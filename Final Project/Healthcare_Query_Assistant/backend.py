import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipelines.rag_pipeline import get_rag_chain

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

    # 2. Initialize SQL Database & Agent
    db = SQLDatabase.from_uri("sqlite:///healthcare.db")
    sql_agent = create_sql_agent(
        llm=llm, db=db, agent_type="openai-tools", verbose=False, top_k=10,
        prefix="""You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run.
        The 'patients' table has EXACTLY these columns: Name, Age, Gender, Blood_Type, Medical_Condition, Admission_Date, Doctor, Hospital, Insurance_Provider, Billing_Amount, Room_Number, Admission_Type, Discharge_Date, Medication, Test_Results.
        RULES:
        1. NEVER invent column names. Use ONLY the exact column names listed above.
        2. When querying names or text fields, ALWAYS use LOWER(column_name) = LOWER('search_term') to handle messy casing.
        3. If the user asks for "all patients" without any specific condition, DO NOT query the entire table. Instead, ask the user how many records they would like to retrieve. If they ask for "all patients with a specific condition", query all matching records.
        4. Never query for all columns from a table. Query only the columns needed.
        5. Format your final response as a clear, conversational answer. If returning a list, format it as a Markdown table."""
    )

    # Safety Patch for SQL Agent
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

    # 3. Initialize RAG Pipeline
    rag_chain, retriever = get_rag_chain(llm)

    # 4. Initialize Orchestrator (Router) with History Awareness
    router_prompt = ChatPromptTemplate.from_template(
        """You are an expert routing agent. Analyze the user's query AND the conversation history to determine which system should handle it.
        
        Conversation History:
        {history}
        
        Current User Query: {query}
        
        - Output 'SQL' if the query is about patient records, medical conditions, billing, doctors, specific patient data, OR if it is a short follow-up command (e.g., 'list 20', 'show 10', 'tell me more') related to a previous database query.
        - Output 'RAG' if the query is about hospital rules, policies, procedures, insurance approvals, or general guidelines.
        - Output 'UNKNOWN' if the query is completely unrelated to healthcare.
        
        Output only 'SQL', 'RAG', or 'UNKNOWN':"""
    )
    router_chain = router_prompt | llm | StrOutputParser()

    return sql_agent, rag_chain, retriever, router_chain

