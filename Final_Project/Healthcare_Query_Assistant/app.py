import streamlit as st
from backend import initialize_system
from ui import apply_custom_css, render_sidebar, render_chat_interface

# 1. Page Config
st.set_page_config(page_title="Healthcare Query Assistant", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# 2. Apply UI Styling
apply_custom_css()

# 3. Render Sidebar
render_sidebar()

# 4. Main Dashboard Header & Metrics
st.title("🏥 Healthcare Multi-Agent Assistant")
st.markdown("### *Query patient records and hospital policies using natural language.*")

col1, col2 = st.columns(2)
col1.metric("Total Patients", "10,000+", "Synthetic Data")
col2.metric("Active Policies", "5", "Admission, Billing, etc.")
st.markdown("---")

# 5. Initialize Backend Systems
try:
    sql_agent, rag_chain, retriever, router_chain = initialize_system()
except ValueError as e:
    st.error(str(e))
    st.stop()

# 6. Render Chat Interface
render_chat_interface(sql_agent, rag_chain, retriever, router_chain)
