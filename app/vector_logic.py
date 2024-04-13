from chromadb import PersistentClient
from app.utils import extract_text_from_docx
from app.constants import DB_PATH
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

def init_chromadb(collection_name, embedding_model, file_path, verbose=False):
    """
    Initializes ChromaDB vectorstore for the given file.

    Args:
        collection_name (str): Name of the collection.
        embedding_model (OpenAIEmbeddings): An instance of the OpenAIEmbeddings object.
        file_path (str): Path to the uploaded file.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose:
        print("Initialize Chromadb")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2800,
                                                   chunk_overlap=280)
    document_text = extract_text_from_docx(file_path)
    document_chunks = text_splitter.split_documents(document_text)
    if verbose:
        print("Amount of chunks for current document: ", len(document_chunks))

    # Init the persistent client to store the documents in the database
    persistent_client = PersistentClient(path=DB_PATH)
    # Init the Chroma client to store documents in the database
    vectorstore = Chroma(collection_name=collection_name,
                         embedding_function=embedding_model,
                         client=persistent_client)
    
    vectorstore.add_documents(documents=document_chunks)
    if verbose:
        print(f"Collection {collection_name} created.")

def setup_chroma_retriever(collection_name, embedding_model, verbose=False):
    """
    Sets up the ChromaDB retriever for the given collection.

    Args:
        collection_name (str): Name of the collection.
        embedding_model (OpenAIEmbeddings): An instance of the OpenAIEmbeddings object.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Retriever: Retriever object for querying documents.
    """
    if verbose:
        print("Setting up ChromaDB retriever for collection: ", collection_name)
        print("Database directory: ", DB_PATH)

    # Init the Chroma client to retrieve documents for the given collection
    langchain_chroma = Chroma(collection_name=collection_name,
                              embedding_function=embedding_model,
                              persist_directory=DB_PATH)
    # Initialize the retriever with cosine similarity and return the top 2 most
    # relevant text chunks of the document
    retriever = langchain_chroma.as_retriever(search_type="similarity",
                                              search_kwargs={"k": 2})
    return retriever

def check_collection_exists(collection_name):
    """
    Checks if the specified collection exists in the database.

    Args:
        collection_name (str): Name of the collection.

    Returns:
        dict: Dictionary with a "status" key. If the collection exists,
              "status" is "success".
              If the collection doesn't exist, "status" is "Failure" and an
              additional "error_message" key provides information about the error.
    """
    try:
        persistent_client = PersistentClient(path=DB_PATH)
        persistent_client.get_collection(collection_name)
        return {"status": "success"}
    except ValueError as e:
        return {
            "status": "Failure",
            "error_message": f"Collection {collection_name} doesn't exist."
        }