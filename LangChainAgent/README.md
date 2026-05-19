```bash
pip install langchain langchain-openai faiss-cpu sentence-transformers
export OPENAI_API_KEY="Individual API key"
python simple_LangChainAgent.py
```

RAG
    - Thought: I need to find invoice INV-001 in the documents
    - Action: search_documents
    - Action Input: invoice INV-001 payment status
    - Observation: Invoice INV-001 from Vendor ABC for 1500 EUR. Status: unpaid.
    - Thought: I have the answer
    - Final Answer: Invoice INV-001 is unpaid.
