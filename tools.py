import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool


# Load environment values before reading keys.
load_dotenv(override=True)


def get_tavily_client():
    """Create the Tavily client only when a search is requested."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to the project's .env file."
        )

    return TavilyClient(api_key=api_key)


@tool
def tavily_search(query: str) -> str:
    """Search the web for recent and reliable information. Returns titles, URLs and snippets."""
    client = get_tavily_client()
    result = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
    )

    output = []
    for item in result.get("results", []):
        output.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {item.get('content', '')[:500]}\n"
        )

    return "\n-----\n".join(output) if output else "No search results were found."


@tool
def scrap_url(url: str) -> str:
    """Scrape a public HTTP/HTTPS URL and return clean text content for deeper reading."""
    try:
        if not url.startswith(("http://", "https://")):
            return "Could not scrape URL: only HTTP and HTTPS URLs are supported."

        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchMind/1.0)"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:6000] if text else "The page did not contain readable text."

    except requests.RequestException as exc:
        return f"Could not scrape URL because the website request failed: {exc}"
    except Exception as exc:
        return f"Could not scrape URL: {exc}"
