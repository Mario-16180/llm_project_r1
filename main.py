import streamlit as st

from os.path import join, exists
import faiss
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import retrieval_qa
from consts import PDFS_DIRECTORY, FAISS_INDEX_PATH
from src.RAG.pdf_rag import (
    upload_pdf,
    load_pdf,
    split_text,
    index_documents,
    retrieve_documents,
    answer_question,
    save_faiss_index,
    load_faiss_index,
)


def chatbot(
    vector_store: InMemoryVectorStore,
    model: OllamaLLM,
    template: str,
):
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


def chatbot_in_vscode(
    # vector_store: InMemoryVectorStore,
    model: OllamaLLM,
    template: str,
):
    documents = load_pdf(
        join(PDFS_DIRECTORY, r"2. Fiebre Tifoidea (Salmonella typhi y paratyphi).pdf")
    )
    documents = split_text(documents)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("data/faiss_index")

    # Retrieval
    vector_store = FAISS.load_local(
        "data/faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    # retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    question = input("Pregunta: ")
    if question:
        related_documents = retrieve_documents(question, vector_store)
        answer = answer_question(question, related_documents, template, model)
        print(answer)

    # index_documents(documents, vector_store)
    # question = input("Pregunta: ")
    # if question:
    #     related_documents = retrieve_documents(question, vector_store)
    #     answer = answer_question(question, related_documents, template, model)
    #     print(answer)


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
    # if exists(FAISS_INDEX_PATH):
    #     vector_store = load_faiss_index(FAISS_INDEX_PATH, embeddings)
    # else:
    #     index = faiss.IndexFlatL2(1000)
    #     vector_store = FAISS(
    #         embedding_function=embeddings,
    #         index=index,
    #         docstore=InMemoryDocstore(),
    #         index_to_docstore_id={},
    #     )
    # vector_store = FAISS(
    #     embedding_function=embeddings,
    #     index="Flat",
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )
    # vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model="deepseek-r1:7b")

    template = """Eres un médico experto en tomar el examen de especialidad en México, el ENARM, y también eres un
    asistente para resolver dudas médicas puntuales. Usa los siguientes fragmentos de contexto recuperados para
    responder la pregunta. Si no sabes la respuesta, simplemente di que no sabes. Recuerda ser conciso y preciso.
    Pregunta: {question}
    Contexto: {context}
    Respuesta:
    """

    # chatbot(embeddings, vector_store, model, template)
    # chatbot_in_vscode(vector_store, model, template)
    chatbot_in_vscode(model, template)
