from tavily import TavilyClient
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
load_dotenv()

@tool
def web_search(query: str) -> str:
    """
    Perform a web search using the Tavily API.

    Args:
        - query (str): The search query.

    Returns:
        - The search results.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query)
    return response