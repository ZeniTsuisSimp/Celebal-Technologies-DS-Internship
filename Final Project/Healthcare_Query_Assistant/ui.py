import streamlit as st
import logging

def apply_custom_css():
    """Applies the dark theme custom CSS."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .stChatInput > div > div > input { background-color: #334155 !important; color: #f8fafc !important; border: 1px solid #475569 !important; border-radius: 20px; padding: 15px 20px; font-size: 16px; }
        .user-message { background-color: #1e3a8a; color: #e2e8f0; padding: 15px; border-radius: 15px; margin-bottom: 10px; border: 1px solid #1e40af; }
        .bot-message { background-color: #1e293b; color: #f8fafc; padding: 15px; border-radius: 15px; margin-bottom: 10px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        section[data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #334155; }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        div[data-testid="stMetric"] { background-color: #1e293b !important; padding: 15px; border-radius: 10px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        div[data-testid="stMetricValue"] { color: #f8fafc !important; }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        div[data-testid="stInfo"] { background-color: #1e3a8a !important; color: #e2e8f0 !important; border: 1px solid #1e40af !important; }
        div[data-testid="stSuccess"] { background-color: #064e3b !important; color: #e2e8f0 !important; border: 1px solid #065f46 !important; }
        h1, h2, h3, h4, h5, h6, p, span, label { color: #f8fafc !important; }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Renders the sidebar controls."""
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        st.subheader(" System Status")
        st.success(" Groq API: Connected")
        st.success("🟢 SQL Database: Active")
        st.success("🟢 Vector Store: Loaded")
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **RAG-Based Healthcare Query Assistant**
        Built for Celebal Technologies.
        - **LLM:** Groq (Llama-3.3-70b)
        - **Framework:** LangChain
        - **Vector DB:** FAISS
        - **Embeddings:** HuggingFace (Local)
        """)

def render_chat_interface(sql_agent, rag_chain, retriever, router_chain):
    """Renders the main chat interface and handles user input with conversation memory."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Helper function to extract recent chat history for context
    def get_chat_history():
        # Grab the last 4 messages (2 pairs of Q&A) to keep context relevant but short
        recent_messages = st.session_state.messages[-4:]
        if not recent_messages:
            return "No previous conversation."
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages])

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about patient records or hospital policies..."):
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Analyzing intent and routing query..."):
                
                # 1. Get the conversation history
                history = get_chat_history()
                
                # 2. Route the query WITH history context
                route = router_chain.invoke({"query": prompt, "history": history}).strip().upper()
                logging.info(f"ROUTING: {route} | QUERY: {prompt}")
                
                # 3. Enrich the prompt with history for the Agents
                enriched_prompt = f"Conversation History:\n{history}\n\nCurrent Question: {prompt}"
                
                # 4. Execute the appropriate agent
                if "SQL" in route:
                    st.info("🔀 Routed to: **NLP-to-SQL Agent** (Patient Database)")
                    try:
                        # Pass the enriched prompt so the SQL agent knows the context
                        response = sql_agent.invoke({"input": enriched_prompt})["output"]
                    except Exception as e:
                        response = f"Sorry, I encountered an error querying the database: {str(e)}"
                        
                elif "RAG" in route:
                    st.info("🔀 Routed to: **RAG Agent** (Policy Documents)")
                    try:
                        docs = retriever.invoke(prompt)
                        if not docs:
                            response = "I couldn't find any relevant hospital policies for that question."
                        else:
                            context = "\n\n".join([doc.page_content for doc in docs])
                            # Pass the enriched prompt to the RAG chain
                            response = rag_chain.invoke({"context": context, "question": enriched_prompt})
                    except Exception as e:
                        response = f"Sorry, I encountered an error retrieving policies: {str(e)}"
                else:
                    response = "I'm not sure how to route that query. I can only answer questions about patient data or hospital policies."
                    logging.info(f"FALLBACK: UNKNOWN | QUERY: {prompt}")

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
