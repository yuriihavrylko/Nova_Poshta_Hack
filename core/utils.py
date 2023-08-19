import glob
import os

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


def load_documents(dirpath):
    documents = []
    for filepath in glob.glob(os.path.join(dirpath, '*txt')):
        documents.extend(TextLoader(filepath, encoding="utf-8").load())
    return documents


def load_texts(dirpath):
    documents = load_documents(dirpath)
    return [doc.page_content for doc in documents]


def create_knowledge_vectordb(dirpath, embedder, chroma_client, chroma_persist_directory):
    collection_names = []
    for folder in glob.glob(os.path.join(dirpath, "*")):
        documents = load_documents(folder)
        db = Chroma.from_documents(documents=documents, embedding=embedder, collection_name=os.path.basename(folder),
                                   client=chroma_client, persist_directory=chroma_persist_directory)
        collection_names.append(db._collection.name)
    return collection_names
