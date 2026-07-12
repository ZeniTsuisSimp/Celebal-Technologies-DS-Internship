# pipelines/sql_pipeline.py
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

def get_sql_agent(llm: ChatOpenAI, db_path: str = "sqlite:///healthcare.db"):
    """Initializes and returns the NLP-to-SQL Agent."""
    
    # Load Database
    db = SQLDatabase.from_uri(db_path)
    
    # Create Agent
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        top_k=10,
        prefix="""You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run.
        
        The 'patients' table has EXACTLY these columns: 
        Name, Age, Gender, Blood_Type, Medical_Condition, Admission_Date, Doctor, Hospital, Insurance_Provider, Billing_Amount, Room_Number, Admission_Type, Discharge_Date, Medication, Test_Results.
        
        RULES:
        1. NEVER invent column names. Use ONLY the exact column names listed above.
        2. When querying names or text fields, ALWAYS use LOWER(column_name) = LOWER('search_term') to handle messy casing.
        3. If the user asks for "all patients" without any specific condition, DO NOT query the entire table. Instead, ask the user how many records they would like to retrieve. If they ask for "all patients with a specific condition", query all matching records.
        4. Never query for all columns from a table. Query only the columns needed.
        5. Format your final response as a clear, conversational answer. If returning a list, format it as a Markdown table."""
    )

    # 🛡️ Safety Patch: Prevent empty string and context overflow errors
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

    return sql_agent