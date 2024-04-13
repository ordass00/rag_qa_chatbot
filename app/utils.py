
import os
from langchain_community.document_loaders import Docx2txtLoader

def create_docs_directory(docs_path):
    """Creates the document folder if it does not exist."""
    if not os.path.exists(docs_path):
        os.mkdir(docs_path)   

def extract_text_from_docx(file_path):
    """Extracts text from a docx file."""
    __, ext = os.path.splitext(file_path)
    if ext == ".docx":
        document_text = Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return document_text

def prepare_collection_name(filename):
    """Prepares the collection name from the filename."""
    collection_name, _ = os.path.splitext(filename)
    collection_name = collection_name.replace(" ", "_")
    return collection_name