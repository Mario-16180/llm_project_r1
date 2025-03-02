from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from os.path import join
from consts import PDFS_DIRECTORY


def upload_pdf(file):
    with open(join(PDFS_DIRECTORY, file.name), "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    """This function loads a PDF file and returns a list of documents.

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    documents = PDFPlumberLoader(file_path).load()
    return documents


def split_text(documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    ).split_documents(documents)


def index_documents(documents, vector_store):
    vector_store.add_documents(documents)


def retrieve_documents(query, vector_store):
    return vector_store.similarity_search(query)


def answer_question(question, documents, template, model):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chained_prompt = prompt | model

    return chained_prompt.invoke({"question": question, "context": context})
