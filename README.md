Car Rental Multi-Agent Assistant (LangChain + LangGraph)

This project is a multi-agent car rental assistant built with LangChain, LangGraph, and Tavily Search.  
Given a natural-language request (dates, cities, pickup/dropoff, budget, car type, etc.), it:

1. Understands and structures the request  
2. Validates constraints and builds a search query  
3. Searches the web for rental options  
4. Filters and ranks results by budget and preferences  
5. Returns a clear, friendly summary of the best options

Everything lives in a single file: `car_rental.py`.


Agent Architecture

The LangGraph workflow uses the following agents:

1. Understanding Agent  
   - Uses an LLM to parse raw user text into a structured `RentalRequest` (Pydantic model).  
   - Extracts pickup/dropoff locations, dates, budget, car type, passengers, etc.

2. Constraint & Query Agent  
   - Validates required fields (e.g., dates, locations).  
   - Builds a web search query string that rental sites/search engines can understand.

3. Web Search & Scraper Agent  
   - Uses `TavilySearchResults` to search the web.  
   - Converts raw results into `RentalOption` objects (vendor, title, link, etc.).  
   - This is the place to plug in real-world price scraping if you want to extend it.

4. Filter & Ranking Agent  
   - Filters options based on total budget (if provided).  
   - Scores and sorts options, keeping the most relevant ones as `top_options`.

5. Answer Agent
   - Uses the LLM again to turn the ranked options + request details  
     into a human-readable answer (bullets, reasoning, and links).

State is passed between nodes using a `GraphState` `TypedDict` that includes:
- `user_input`
- `request` (`RentalRequest`)
- `search_query`
- `options`
- `top_options`
- `answer`

Create a .env file in the project root which consist of the Chatgpt and tavily API key 

Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install langchain-openai langgraph langchain-community \
            tavily-python duckduckgo-search requests beautifulsoup4 \
            pydantic python-dotenv


