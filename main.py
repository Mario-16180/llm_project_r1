import streamlit as st

from os.path import join

from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from consts import PDFS_DIRECTORY
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
    vector_store = FAISS(
        embedding_function=embeddings,
        index="Flat",
        docstore=InMemoryDocstore,
        index_to_docstore_id={},
    )
    # vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model="deepseek-r1:7b")

    template = """Eres un médico experto en tomar el examen de especialidad en México, el ENARM, y también eres un
    asistente para resolver dudas médicas puntuales. Usa los siguientes fragmentos de contexto recuperados para
    responder la pregunta. Si no sabes la respuesta, simplemente di que no sabes. Recuerda ser conciso y preciso.
    Pregunta: {question}
    Contexto: {context}
    Respuesta:
    """

    uploaded_file = st.file_uploader("Sube el archivo PDF.", type=["pdf"])

    if uploaded_file:
        upload_pdf(uploaded_file)
        documents = load_pdf(join(PDFS_DIRECTORY, uploaded_file.name))
        documents = split_text(documents)
        index_documents(documents, vector_store)

        question = st.chat_input()
        if question:
            st.chat_message("user").write(question)
            related_documents = retrieve_documents(question, vector_store)
            answer = answer_question(question, related_documents, template, model)
            st.chat_message("bot").write(answer)


if __name__ == "__main__":
    chatbot()
