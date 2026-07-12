from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(llm: ChatOpenAI, faiss_path: str = "faiss_index"):
    """Initializes and returns the RAG Pipeline."""
    
    # Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Define RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template(
        """You are a helpful hospital policy assistant. Answer the user's question based ONLY on the following context. 
        If the answer is not in the context, say "I don't have that information in the policy documents."
        
        Context: {context}
        Question: {question}
        """
    )
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    return rag_chain, retriever