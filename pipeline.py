from agent import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)
from error_handling import safe_invoke


def run_research_pipeline(topic: str) -> dict:
    """
    Run Search -> Reader -> Writer -> Critic.
    This module can be used independently; the Streamlit UI in app.py
    runs the same stages inline so it can update its progress cards.
    """
    if not topic or not topic.strip():
        raise ValueError("Please provide a research topic.")

    state = {}

    print("\n" + "=" * 60)
    print("STEP 1 - Search Agent")
    print("=" * 60)

    search_agent = build_search_agent()
    search_result = safe_invoke(
        search_agent,
        {
            "messages": [
                ("user", f"Find recent, reliable and detailed information about: {topic}")
            ]
        },
        "Search Agent",
    )
    state["search_results"] = search_result["messages"][-1].content

    print("\n" + "=" * 60)
    print("STEP 2 - Reader Agent")
    print("=" * 60)

    reader_agent = build_reader_agent()
    reader_result = safe_invoke(
        reader_agent,
        {
            "messages": [
                (
                    "user",
                    f"Based on the following search results about '{topic}', "
                    "pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}",
                )
            ]
        },
        "Reader Agent",
    )
    state["scraped_content"] = reader_result["messages"][-1].content

    print("\n" + "=" * 60)
    print("STEP 3 - Writer Chain")
    print("=" * 60)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = safe_invoke(
        writer_chain,
        {"topic": topic, "research": research_combined},
        "Report Generation",
    )

    print("\n" + "=" * 60)
    print("STEP 4 - Critic Chain")
    print("=" * 60)

    state["feedback"] = safe_invoke(
        critic_chain,
        {"report": state["report"]},
        "Critic Review",
    )

    return state


if __name__ == "__main__":
    research_topic = input("Enter a research topic: ").strip()
    output = run_research_pipeline(research_topic)
    print("\nResearch pipeline completed successfully.")
    print(output["report"])
