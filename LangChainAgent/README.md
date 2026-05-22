### Simple LangChain Agent

This project demonstrates a simple AI agent built using  LangChain agent and Python. The goal of the project is to explore how Large Language Models (LLMs) can interact with tools, process user prompts, and perform basic reasoning and task execution in an agent-based workflow.

The project includes:
- Building a simple LangChain agent architecture
- Prompt handling and conversational interaction
- Integration with external tools and APIs
- LLM-based reasoning and response generation
- Basic agent workflow and execution pipeline

This project provides a hands-on introduction to AI agents and demonstrates how LangChain can be used to create intelligent applications capable of automating tasks and interacting dynamically with users.

Technologies used include Python, LangChain, and LLM-based agent frameworks.


```bash
pip install langchain langchain-openai faiss-cpu sentence-transformers
export OPENAI_API_KEY="Individual API key"
python simple_LangChainAgent.py
```


#### Prompt engineering and RAG

    - Thought: I need to find invoice INV-001 in the documents

    - Action: search_documents

    - Action Input: invoice INV-001 payment status

    - Observation: Invoice INV-001 from Vendor ABC for 1500 EUR. Status: unpaid.

    - Thought: I have the answer

    - Final Answer: Invoice INV-001 is unpaid.
