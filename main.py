import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.RAG.pdf_rag import (
    upload_pdf,
    load_pdf,
    split_text,
    index_documents,
    retrieve_documents,
    answer_question,
)


def chatbot():
    embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
    vector_store = FAISS(embeddings=embeddings, index_path="data/faiss_index")
    model = OllamaLLM(model="deepseek-r1:7b")

    template = """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Remember to be concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        upload_pdf(uploaded_file)
        documents = load_pdf(uploaded_file.name)
        documents = split_text(documents)
        index_documents(documents, vector_store)

        question = st.text_input("Ask a question")
        if question:
            retrieved_documents = retrieve_documents(question, vector_store)
            answer = answer_question(question, retrieved_documents, template, model)
            st.write(answer)


if __name__ == "__main__":
    chatbot()
