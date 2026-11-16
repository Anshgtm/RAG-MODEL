
import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# Web Search Tool
@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.
    Returns titles, URLs and snippets."""

    result = tavily_client.search(
        query=query,
        max_results=5
    )

    out = []

    for r in result["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}\n"
        )

    return "\n---\n".join(out)


# Web Scraping Tool
@tool
def web_scrape(url: str) -> str:
    """Scrape a webpage URL and return its clean text content."""

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(
            separator=" ",
            strip=True
        )[:3000]

    except requests.RequestException as e:
        return f"Could not scrape URL: {str(e)}"
    except Exception as e:
        return f"Unexpected scraping error: {str(e)}"
