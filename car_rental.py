"""
Car Rental Multi-Agent System (LangChain + LangGraph) - Single File

Agents:
1. Understanding Agent         -> parses user text into RentalRequest
2. Constraint & Query Agent    -> validates & builds search query
3. Web Search & Scraper Agent  -> DuckDuckGo + BeautifulSoup
4. Filter & Ranking Agent      -> filters by total budget & ranks
5. Answer Agent                -> formats final response

Requirements (pip):
- langchain-openai
- langgraph
- duckduckgo-search
- requests
- beautifulsoup4
- pydantic

Also set:  export OPENAI_API_KEY="your_key_here"
"""

from __future__ import annotations

from dotenv import load_dotenv
import os

import json
from datetime import date
from typing import Optional, List, TypedDict

import requests
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


# Load the .env file
load_dotenv()

# Now OPENAI_API_KEY is available to LangChain & OpenAI SDK
# No further logic changes needed!


# ============================
#   Pydantic Models
# ============================

class RentalRequest(BaseModel):
    raw_text: str
    intent: str = "car_rental_search"

    pickup_city: Optional[str] = None
    dropoff_city: Optional[str] = None
    pickup_location: Optional[str] = None
    dropoff_location: Optional[str] = None

    pickup_date: Optional[date] = None
    dropoff_date: Optional[date] = None

    # Only total budget (whole trip), no per-day budget
    total_budget: Optional[float] = None


class RentalOption(BaseModel):
    vendor: str
    car_name: str
    total_price: float
    pickup_location: str
    dropoff_location: str
    rating: Optional[float]
    link: str
    score: Optional[float] = None


# ============================
#   LangGraph State
# ============================

class GraphState(TypedDict, total=False):
    user_input: str
    request: RentalRequest
    search_query: str
    options: List[RentalOption]
    top_options: List[RentalOption]
    answer: str


# ============================
#   LLM Helper
# ============================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Call ChatGPT (via LangChain) and parse JSON from the response.
    We instruct it to return ONLY a JSON object.
    """
    msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = msg.content

    # Extract JSON substring in case model adds extra text
    start = content.find("{")
    end = content.rfind("}")
    json_str = content[start : end + 1]
    return json.loads(json_str)


# ============================
#   Agent 1: Understanding Agent
# ============================

def understanding_agent(state: GraphState) -> GraphState:
    """
    Takes raw user_input and turns it into a RentalRequest.
    Uses the LLM to extract: pickup/dropoff locations, dates, total_budget.
    """

    user_text = state["user_input"]

    system = """
You are an information extraction assistant for a car rental system.
Given a user's free-text request, extract:

- pickup_city (city name or null)
- pickup_location (more specific place like 'LAX airport' or 'downtown' or null)
- dropoff_city
- dropoff_location
- pickup_date (ISO YYYY-MM-DD or null)
- dropoff_date (ISO YYYY-MM-DD or null)
- total_budget (number in USD for the whole rental, or null)

Respond ONLY as a JSON object with these exact keys.
If something isn't clearly stated, use null.
"""
    user = f"User request:\n{user_text}"

    data = call_llm_json(system, user)

    # Safely parse dates
    def parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except Exception:
            return None

    pickup_date = parse_date(data.get("pickup_date"))
    dropoff_date = parse_date(data.get("dropoff_date"))

    total_budget_raw = data.get("total_budget")
    try:
        total_budget = float(total_budget_raw) if total_budget_raw is not None else None
    except Exception:
        total_budget = None

    req = RentalRequest(
        raw_text=user_text,
        pickup_city=data.get("pickup_city"),
        pickup_location=data.get("pickup_location"),
        dropoff_city=data.get("dropoff_city"),
        dropoff_location=data.get("dropoff_location"),
        pickup_date=pickup_date,
        dropoff_date=dropoff_date,
        total_budget=total_budget,
    )

    state["request"] = req
    return state


# ============================
#   Agent 2: Constraint & Query Agent
# ============================

def constraint_agent(state: GraphState) -> GraphState:
    """
    Validates basic constraints (e.g., dropoff after pickup),
    and builds a search query string using the RentalRequest.
    """
    req = state["request"]

    # Basic date sanity check
    if req.pickup_date and req.dropoff_date:
        if req.dropoff_date <= req.pickup_date:
            raise ValueError("Dropoff date must be after pickup date")

    parts = ["car rental"]

    if req.pickup_city:
        parts.append(req.pickup_city)

    if req.pickup_date and req.dropoff_date:
        parts.append(f"{req.pickup_date} to {req.dropoff_date}")

    if req.total_budget is not None:
        parts.append(f"total under {int(req.total_budget)} USD")

    query = " ".join(parts)
    state["search_query"] = query
    return state


# ============================
#   Agent 3: Web Search & Scraper Agent
# ============================


tavily = TavilySearchResults(max_results=5)

def search_agent(state: GraphState) -> GraphState:
    req = state["request"]
    query = state["search_query"]

    raw_results = tavily.run(query)
    options: List[RentalOption] = []

    for r in raw_results:
        url = r.get("url")
        title = r.get("title") or "Car Rental Option"

        option = RentalOption(
            vendor=r.get("source", "Web"),
            car_name=title[:60],
            total_price=150.0,  # still dummy until you parse real prices
            pickup_location=req.pickup_location or (req.pickup_city or "Unknown"),
            dropoff_location=req.dropoff_location or (req.dropoff_city or "Unknown"),
            rating=4.0,
            link=url,
        )
        options.append(option)

    state["options"] = options
    return state


# ============================
#   Agent 4: Filter & Ranking Agent
# ============================

def ranking_agent(state: GraphState) -> GraphState:
    """
    Filters options by total_budget (if provided),
    then scores and sorts them.
    """
    req = state["request"]
    options = state.get("options", [])

    # Filter by total_budget if set
    filtered: List[RentalOption] = []
    for opt in options:
        if req.total_budget is not None and opt.total_price > req.total_budget:
            continue
        filtered.append(opt)

    # Score by price + rating
    for opt in filtered:
        price_score = 1.0 / (opt.total_price + 1e-6)
        rating_score = (opt.rating or 3.5) / 5.0
        opt.score = 0.6 * price_score + 0.4 * rating_score

    filtered.sort(key=lambda o: o.score or 0.0, reverse=True)
    state["top_options"] = filtered[:5]
    return state


# ============================
#   Agent 5: Answer Agent
# ============================

def answer_agent(state: GraphState) -> GraphState:
    """
    Formats the top options into a human-friendly answer string.
    """
    options = state.get("top_options", [])
    req = state["request"]

    if not options:
        state["answer"] = (
            "I couldn’t find good rental options that clearly match your criteria. "
            "You might try adjusting your dates, location, or budget."
        )
        return state

    lines = []
    lines.append("Here are some car rental options based on your request:\n")

    if req.pickup_city or req.pickup_date:
        details_bits = []
        if req.pickup_city:
            details_bits.append(f"city: {req.pickup_city}")
        if req.pickup_date and req.dropoff_date:
            details_bits.append(f"dates: {req.pickup_date} to {req.dropoff_date}")
        if req.total_budget is not None:
            details_bits.append(f"budget ≤ ${int(req.total_budget)} total")
        if details_bits:
            lines.append("Filters used: " + ", ".join(details_bits) + "\n")

    for i, opt in enumerate(options, start=1):
        lines.append(
            f"{i}. {opt.vendor} – {opt.car_name}\n"
            f"   Estimated total: ${opt.total_price:.0f}\n"
            f"   Pickup: {opt.pickup_location}\n"
            f"   Dropoff: {opt.dropoff_location}\n"
            f"   Link: {opt.link}\n"
        )

    state["answer"] = "\n".join(lines)
    return state


# ============================
#   LangGraph Definition
# ============================

def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("understand", understanding_agent)
    graph.add_node("constraints", constraint_agent)
    graph.add_node("search", search_agent)
    graph.add_node("ranking", ranking_agent)
    graph.add_node("answer", answer_agent)

    # Entry point
    graph.set_entry_point("understand")

    # Edges (linear for now)
    graph.add_edge("understand", "constraints")
    graph.add_edge("constraints", "search")
    graph.add_edge("search", "ranking")
    graph.add_edge("ranking", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# ============================
#   Example Usage
# ============================

if __name__ == "__main__":
    workflow = build_graph()

    user_text = (
        "I need a rental car in Los Angeles from 2025-12-05 to 2025-12-08, "
        "pickup at LAX and dropoff near Santa Monica, with a total budget of 250 dollars."
    )

    result = workflow.invoke({"user_input": user_text})

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("answer", "No answer generated."))
