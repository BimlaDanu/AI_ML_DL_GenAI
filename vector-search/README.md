### Project Summary

This project is a simple semantic FAQ search system built using Python, FastAPI, Sentence Transformers, and FAISS.

The main idea behind the project is to search based on meaning instead of exact keywords. User questions are converted into vector embeddings using a transformer model, and those embeddings are stored in a vector database (FAISS). When a user sends a query, the system finds the most semantically similar question and returns the corresponding answer.

I built this project to better understand how modern NLP retrieval systems work internally, especially concepts like embeddings, vector similarity search, and retrieval-based AI systems. It also helped me learn how APIs are designed and tested using FastAPI and Swagger UI.

This project is intentionally lightweight and beginner-friendly, but it reflects the core workflow used in larger AI applications such as semantic search engines, recommendation systems, and Retrieval-Augmented Generation (RAG) pipelines.

### Python Environment and bash command

```python
faq-vector-search/
│
├── app.py
├── data.py
├── requirements.txt
└── README.md
```


```bash
pip install -r requirements.txt # Setting up the environment
```

```bash
uvicorn app:app --reload # running the app.py
Ctrl + C # stopping the app.py
```

- Build a small system where users ask questions and get relevant answers using a vector database.

- User -> API -> Embedding -> Vector DB -> Top match -> Answer

**FastAPI:** A modern Python framework for building fast APIs with automatic validation and documentation.

**Uvicorn:** A lightweight, high-performance server used to run FastAPI applications.

**FAISS:**  A library for efficient similarity search over high-dimensional vectors (used as a vector database).

**sentence-transformers:** A library that converts text into semantic vector embeddings using transformer models.

**Streamlit:** A tool for quickly building interactive web apps and dashboards for machine learning projects.

Querries on web 
    - http://127.0.0.1:8000/docs

    - http://127.0.0.1:8000/search?query=refund

    - http://127.0.0.1:8000/search?query=change email

    - http://127.0.0.1:8000/docs

    - http://127.0.0.1:8000/search?query=I forgot my password