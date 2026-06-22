from langchain.tools import tool
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from rich import print
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search(query: str):
    """ Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    result = tavily.search(query=query,search_depth="basic",max_results = 5)
    out = []
    for r in result['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    return "\n-----\n".join(out)
# print(tavily_search.invoke("what is latest news about wars?"))

@tool
def scrap_url(url:str)->str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
        # print("resp",resp)
        soup = BeautifulSoup(resp.text,"html.parser")
        print("\n")
        print("soup",soup)
        for tag in soup(["script", "style", "nav", "footer"]):
           tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
# print(scrap_url.invoke("https://www.hindustantimes.com/india-news/sanjay-raut-doubles-down-on-abuses-for-rebel-shiv-sena-ubt-uddhav-sena-then-says-dont-cut-101781778624747.html"))