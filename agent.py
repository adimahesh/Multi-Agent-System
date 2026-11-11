import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import tavily_search, scrap_url


# Load .env from the project directory and prefer it over stale Windows
# environment variables. Never print the API key.
load_dotenv(override=True)

api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()

if not api_key:
    raise RuntimeError(
        "Google Gemini API key is missing. Add GOOGLE_API_KEY to your .env file."
    )

# Gemini model used by all four stages.
# You can override it in .env with GEMINI_MODEL=...
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0,
    google_api_key=api_key,
)


# 1. Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[tavily_search],
    )


# 2. Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrap_url],
    )


# 3. Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# 4. Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
