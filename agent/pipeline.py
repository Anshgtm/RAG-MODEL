from agent import search_agent,reader_agent,writer_chain,critic_chain
def research_pipeline(topic:str)->dict:
    state = {}
    searching_agent = search_agent()
    search_result = searching_agent.invoke({
        "messages":[("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    print('\n search result',state["search_results"])

    reading_agent = reader_agent()
    reader_result = reading_agent.invoke({
            "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
})
    state["scraped_content"] = reader_result["messages"][-1].content
    print('\n reader result', state["scraped_content"])

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n", state['report'])


    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    research_pipeline(topic)


