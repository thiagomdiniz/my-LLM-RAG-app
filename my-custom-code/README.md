## my-custom-code

```
pip3 install -r requirements.txt
cp .env.example .env
```

### Document Summarization using LLMs

`python3 summarize.py`

Reference: https://python.langchain.com/docs/use_cases/summarization

### Q&A with RAG

Question-answering (Q&A) chatbot using the Retrieval Augmented Generation technique.

You can use `Pinecone` or `Chroma` as a vector database, depending on the `.env` file settings.  
You can use local `OpenAI` or local `Meta Llama 2` as LLM, depending on the `.env` file settings.

`python3 app.py`

Reference: https://python.langchain.com/docs/use_cases/question_answering/