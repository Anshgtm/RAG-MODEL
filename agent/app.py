import os
import re
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Load local .env variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Deep Research AI | Multi-Agent RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: 800;
        color: #2563EB;
        text-align: center;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 6px;
    }
    .agent-step {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #F1F5F9;
        margin-bottom: 8px;
        border-left: 4px solid #3B82F6;
    }
    .source-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = ""

# Sidebar Configurations
with st.sidebar:
    st.title("⚙️ Agent Settings")
    st.caption("Configure multi-agent parameters and credentials")

    # API Keys Section
    with st.expander("🔑 API Credentials", expanded=False):
        groq_env = os.getenv("GROQ_API_KEY", "")
        tavily_env = os.getenv("TAVILY_API_KEY", "")

        groq_api_key = st.text_input(
            "Groq API Key",
            value=groq_env,
            type="password",
            help="Required for LLM inferences (Groq)"
        )
        tavily_api_key = st.text_input(
            "Tavily API Key",
            value=tavily_env,
            type="password",
            help="Required for Tavily Web Search"
        )

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key

        if groq_api_key and tavily_api_key:
            st.success("✅ Keys loaded successfully")
        else:
            st.warning("⚠️ Missing keys in .env or input")

    # Model Configuration
    st.subheader("🧠 Model Selection")
    model_choices = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "Custom"
    ]
    selected_model = st.selectbox("Select LLM Model", model_choices, index=0)
    if selected_model == "Custom":
        model_name = st.text_input("Custom Model ID", value="openai/gpt-oss-120b")
    else:
        model_name = selected_model

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    st.divider()

    # Agent Architecture Info
    st.subheader("🤖 Pipeline Flow")
    st.markdown("""
    1. **🔍 Search Agent**
       - Query Tavily for real-time web results.
    2. **🌐 Reader Agent**
       - Autonomous scraping of key source URLs.
    3. **✍️ Writer Agent**
       - Synthesizes findings into a structured report.
    4. **⚖️ Critic Agent**
       - Rigorous evaluation, score & improvement feedback.
    """)

    st.divider()

    # Research History
    st.subheader("📚 Saved Researches")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history)):
            btn_label = f"{item['topic'][:24]}... ({item['timestamp']})"
            if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.current_result = item
                st.session_state.selected_topic = item["topic"]
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()
    else:
        st.caption("No past research in this session.")

# Function to run the research pipeline with real-time UI status updates
def run_research(topic: str, model_id: str, temp: float):
    """Executes the agentic research pipeline with granular progress tracking."""
    import agent
    from tools import tavily_client

    # Update model settings if modified
    try:
        from langchain_groq import ChatGroq
        custom_llm = ChatGroq(model=model_id, temperature=temp)
        agent.llm = custom_llm
        agent.writer_chain = agent.writter_prompt | custom_llm | agent.StrOutputParser()
        agent.critic_chain = agent.critic_prompt | custom_llm | agent.StrOutputParser()
    except Exception as e:
        st.warning(f"Using default agent LLM settings: {e}")

    state = {}
    
    with st.status("🚀 Running Autonomous Research Pipeline...", expanded=True) as status:
        # Step 1: Web Search
        status.update(label="🔍 [1/4] Search Agent is querying the web...", state="running")
        st.write("🔎 Querying Tavily for recent, reliable, and verified data...")
        searching_agent = agent.search_agent()
        search_result = searching_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
        st.success("✅ Search completed with key sources and snippets.")

        # Step 2: Reader / Web Scraper
        status.update(label="🌐 [2/4] Reader Agent is selecting and scraping deep sources...", state="running")
        st.write("📖 Selecting the most relevant URL and extracting full page content...")
        reading_agent = agent.reader_agent()
        reader_result = reading_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
        st.success("✅ Deep scraping completed.")

        # Step 3: Writer Agent
        status.update(label="✍️ [3/4] Writer Agent is synthesizing the research report...", state="running")
        st.write("📝 Structuring report: Introduction, Key Findings, Conclusion, and Sources...")
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = agent.writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })
        st.success("✅ Research report synthesized.")

        # Step 4: Critic Agent
        status.update(label="⚖️ [4/4] Critic Agent is evaluating the output...", state="running")
        st.write("🧐 Evaluating factuality, depth, clarity, and scoring out of 10...")
        state["feedback"] = agent.critic_chain.invoke({
            "report": state["report"]
        })
        st.success("✅ Critical evaluation complete.")

        status.update(label="🎉 Research and evaluation pipeline finished!", state="complete", expanded=False)

    return state

# Helper to extract score from critic feedback
def parse_critic_score(feedback_text: str):
    match = re.search(r"Score:\s*(\d+(\.\d+)?)\s*/\s*10", feedback_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"

# Main Layout
st.markdown('<div class="main-title">🔬 Autonomous Deep Research & Critic Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A multi-agent intelligence system powered by LangChain, Groq, and Tavily for autonomous web search, deep page reading, synthesis, and peer review.</div>', unsafe_allow_html=True)

# Prompt Starters
st.markdown("**💡 Quick Starters:**")
sample_topics = [
    "Latest Breakthroughs in Quantum Computing in 2026",
    "State of Autonomous AI Agents and Multi-Agent Workflows",
    "Recent Advancements in Fusion Energy Commercialization",
    "Next-Generation Solid-State Batteries for Electric Vehicles"
]

cols = st.columns(len(sample_topics))
for idx, s_topic in enumerate(sample_topics):
    if cols[idx].button(s_topic, key=f"sample_{idx}", use_container_width=True):
        st.session_state.selected_topic = s_topic

# Research Input Bar
with st.container():
    topic_input = st.text_input(
        "Enter your research topic:",
        value=st.session_state.selected_topic,
        placeholder="e.g. Advancements in neuromorphic computing architectures",
        help="Specify the subject you want the multi-agent system to investigate."
    )
    
    col_run, col_clear = st.columns([6, 1])
    run_btn = col_run.button("🚀 Start Deep Research", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        st.session_state.selected_topic = ""
        st.session_state.current_result = None
        st.rerun()

# Execution Trigger
if run_btn:
    if not topic_input.strip():
        st.error("⚠️ Please enter a research topic to proceed.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY is missing! Please enter it in the sidebar or provide it in your .env file.")
    elif not os.getenv("TAVILY_API_KEY"):
        st.error("⚠️ TAVILY_API_KEY is missing! Please enter it in the sidebar or provide it in your .env file.")
    else:
        try:
            start_time = time.time()
            result = run_research(topic_input.strip(), model_name, temperature)
            duration = round(time.time() - start_time, 2)
            
            # Save to current state & history
            result["topic"] = topic_input.strip()
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            result["duration"] = duration
            st.session_state.current_result = result
            st.session_state.history.append(result)
            st.toast(f"Research finished in {duration}s!", icon="🎉")
        except Exception as e:
            st.error(f"❌ An error occurred during pipeline execution: {str(e)}")
            st.exception(e)

# Results Display Section
if st.session_state.current_result:
    res = st.session_state.current_result
    st.divider()
    
    # Header summary
    header_col1, header_col2, header_col3 = st.columns([4, 1, 1])
    with header_col1:
        st.subheader(f"📌 Research on: {res.get('topic')}")
        st.caption(f"Completed at: {res.get('timestamp', 'Just now')} | Processing Time: {res.get('duration', 'N/A')}s")
    
    score = parse_critic_score(res.get("feedback", ""))
    with header_col2:
        st.metric(label="Critic Score", value=f"{score}/10" if score != "N/A" else "Evaluated")
    
    with header_col3:
        report_text = res.get("report", "")
        word_count = len(report_text.split())
        st.metric(label="Report Words", value=f"{word_count}")

    # Tabs for structured viewing
    tab_report, tab_critic, tab_sources, tab_raw = st.tabs([
        "📄 Research Report",
        "⚖️ Critic Review & Evaluation",
        "🌐 Search & Scraped Sources",
        "⚙️ Agent Raw State"
    ])

    with tab_report:
        col_rep_top, col_rep_dl = st.columns([5, 1])
        with col_rep_dl:
            st.download_button(
                label="📥 Download .md",
                data=res.get("report", ""),
                file_name=f"research_report_{res.get('topic', 'topic')[:20].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        st.markdown(res.get("report", "*No report generated yet.*"))

    with tab_critic:
        st.subheader("🧐 Peer Review & Evaluation")
        feedback_content = res.get("feedback", "")
        
        # Display formatted feedback
        st.info(feedback_content)

    with tab_sources:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("### 🔍 Search Agent Results")
            st.markdown(res.get("search_results", "No search results recorded."))
        
        with col_s2:
            st.markdown("### 🌐 Reader Agent Scraped Content")
            scraped = res.get("scraped_content", "No scraped content recorded.")
            st.text_area("Scraped Web Text (Truncated Preview)", scraped, height=400)

    with tab_raw:
        st.markdown("### 🛠️ Execution Pipeline Debug State")
        st.json({
            "topic": res.get("topic"),
            "timestamp": res.get("timestamp"),
            "duration": res.get("duration"),
            "search_results_length": len(res.get("search_results", "")),
            "scraped_content_length": len(res.get("scraped_content", "")),
            "report_length": len(res.get("report", "")),
            "feedback": res.get("feedback", "")
        })
