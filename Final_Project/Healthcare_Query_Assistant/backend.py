import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipelines.rag_pipeline import get_rag_chain
import logging

# 🌟 FIX: Get the absolute path to the directory where backend.py is located
BASE_DIR = Path(__file__).resolve().parent

# Setup Logging
logging.basicConfig(filename='agent_routing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_system():
    """Initializes and returns all AI agents and databases."""
    load_dotenv()
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file!")

    # 1. Initialize LLM
    llm = ChatOpenAI(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY, temperature=0)

    # 🌟 FIX: Use absolute path for SQLite database
    db_path = BASE_DIR / "healthcare.db"
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # 2. Initialize SQL Agent (Keep your existing prefix and safe_output wrapper here)
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        top_k=10,
        prefix="""You are a SQL agent for a healthcare database.
        Table 'patients' has columns: Name, Age, Gender, Blood_Type, Medical_Condition, Admission_Date, Doctor, Hospital, Insurance_Provider, Billing_Amount, Room_Number, Admission_Type, Discharge_Date, Medication, Test_Results.
        RULES:
        1. Use exact column names only. Never invent columns.
        2. Use LOWER() for name/text searches to handle messy casing.
        3. If asked for "all patients" without filters, ask user to specify a limit.
        4. Query only needed columns.
        5. Return results as clear, conversational answers with Markdown tables."""
    )

    # Safety Patch (Keep your existing safe_output wrapper code here)
    def safe_output(method):
        def wrapper(*args, **kwargs):
            try:
                result = method(*args, **kwargs)
            except Exception as e:
                return f"Error executing tool: {str(e)}"
            if result is None or result == "" or result == [] or result == ():
                return "No results found."
            if isinstance(result, str) and not result.strip():
                return "No results found."
            result_str = str(result)
            if len(result_str) > 4000:
                return result_str[:4000] + "\n\n[SYSTEM NOTE: The query returned too many rows. Results have been truncated. Please inform the user and suggest they refine their query.]"
            return result_str
        return wrapper

    for tool in sql_agent.tools:
        if hasattr(tool, '_run'):
            tool._run = safe_output(tool._run)

    # 🌟 FIX: Pass the absolute path for the FAISS index
    faiss_path = str(BASE_DIR / "faiss_index")
    rag_chain, retriever = get_rag_chain(llm, faiss_path=faiss_path)

    # 4. Initialize Orchestrator (Router)
    router_prompt = ChatPromptTemplate.from_template(
        """You are an expert routing agent. Analyze the user's query AND the conversation history to determine which system should handle it.
        
        Conversation History:
        {history}
        
        Current User Query: {query}
        
        Routing Rules:
        - Output 'SQL' if the query is about:
          * Patient records, demographics, or personal information
          * Medical conditions, diagnoses, or test results of specific patients
          * Billing amounts, insurance providers, or costs for patients
          * Doctor names, hospital names, room numbers
          * Admission dates, discharge dates for specific patients
          * COUNT, SUM, AVG, or statistical queries about patient data
          * Short follow-up commands (e.g., 'list 20', 'show 10', 'tell me more')
        
        - Output 'RAG' if the query is about:
          * Hospital policies, procedures, rules, or guidelines
          * Discharge policy, admission policy, billing policy
          * Requirements, conditions, or steps for hospital processes
          * Visitor rules, emergency procedures, insurance approval processes
          * "What is the policy for...", "How to...", "What are the requirements..."
        
        - Output 'UNKNOWN' if the query is completely unrelated to healthcare.
        
        Output only 'SQL', 'RAG', or 'UNKNOWN':"""
    )
    router_chain = router_prompt | llm | StrOutputParser()

    return sql_agent, rag_chain, retriever, router_chain