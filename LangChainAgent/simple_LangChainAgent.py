"""
Simple LangChain Agent 
Two tools:
  1. Document search (RAG over a small text store)
  2. Calculator

The agent decides which tool to use based on the question.
"""

# Install first
# pip install langchain langchain-openai faiss-cpu sentence-transformers

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain import hub
import math

# Small document store 
DOCUMENTS = [
    "Invoice INV-001 from Vendor ABC for 1500 EUR dated 01.05.2026. Status: unpaid.",
    "Invoice INV-002 from Vendor XYZ for 3200 EUR dated 10.05.2026. Status: paid.",
    "Invoice INV-003 from Vendor ABC for 800 EUR dated 15.05.2026. Status: pending review.",
    "Vendor ABC has a credit limit of 5000 EUR and payment terms of 30 days.",
    "Vendor XYZ has a credit limit of 10000 EUR and payment terms of 14 days.",
]

#  Build vector store from documents 
def build_vectorstore():
    docs = [Document(page_content=text) for text in DOCUMENTS]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = build_vectorstore()

#  Define Tool 1: Document Search 
@tool
def search_documents(query: str) -> str:
    """
    Search the accounting document store for invoices,
    vendor information, payment status, and related records.
    Use this when the question is about invoices, vendors, or payments.
    """
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "No relevant documents found."
    return "\n".join([doc.page_content for doc in results])

#  Define Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any numerical calculations such as totals,
    percentages, VAT calculations, or budget checks.
    Examples: '1500 + 3200', '3200 * 0.19', '5000 - 1500 - 800'
    """
    try:
        # safe eval — only math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Set up the LLM: os.environ["OPENAI_API_KEY"] = "Individual key"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,          # deterministic — important for agents
    api_key=os.environ.get("OPENAI_API_KEY")
)

#  Load ReAct prompt from LangChain hub; A standard ReAct prompt: Thought -> Action -> Observation loop
prompt = hub.pull("hwchase17/react")

#  Build the agent
tools = [search_documents, calculator]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # shows the full reasoning chain — great for learning
    max_iterations=5,       # prevents infinite loops
    handle_parsing_errors=True
)

#  Test questions
if __name__ == "__main__":

    questions = [
        #Document search tool
        "What is the payment status of invoice INV-001?",

        #Calculator tool
        "What is 19% VAT on 3200 EUR?",

        # Uses BOTH tools - agent must decide order
        "What is the total unpaid amount for Vendor ABC, and does it exceed their credit limit?",

        # Reasoning across both tools
        "How much has Vendor XYZ been paid in total?",
    ]

    for question in questions:
        print("\n" + "="*60)
        print(f"QUESTION: {question}")
        print("="*60)
        result = agent_executor.invoke({"input": question})
        print(f"\nFINAL ANSWER: {result['output']}")
