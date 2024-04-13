import os
import time
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage
from chromadb import PersistentClient
from app.constants import DB_PATH
from app.vector_logic import setup_chroma_retriever, check_collection_exists
from app.chain_logic import generate_question, generate_answer

def init_llm():
    """
    Initializes and returns an instance of the ChatOpenAI and OpenAIEmbeddings
    object using the OpenAI API key retrieved from environment variables.
    Raises an exception if the API key is not found.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI object.
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings object.

    Raises:
        ValueError: If the OpenAI API environment variable is not set.
    """
    try:
        llm_api_key = os.getenv("OPENAI_API_KEY")
        if not llm_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    except ValueError as e:
        st.error("Error occured: " + str(e))
        st.stop()

    llm = ChatOpenAI(api_key=llm_api_key)
    embedding_model = OpenAIEmbeddings(api_key=llm_api_key,
                                       model="text-embedding-3-large")
    return llm, embedding_model

def select_file(filename, llm, embedding_model, verbose=False):
    """
    Selects a file for chatting and initializes the LLM question-answering logic.

    Args:
        filename (str): The name of the file to select.
        llm (ChatOpenAI): An instance of the ChatOpenAI object.
        embedding_model (OpenAIEmbeddings): An instance of the OpenAIEmbeddings object.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose:
        print("Chatting with: ", filename)
    question = llm_qa_logic(st.session_state.collection_name,
                            llm,
                            embedding_model,
                            verbose)

    st.session_state.chat_history.clear()
    st.session_state.chat_history.append(AIMessage(question))

def llm_qa_logic(collection, llm, embedding_model, verbose=False):
    """
    Combines the LLM question-answering logic with the Streamlit
    session state variables.

    Args:
        collection (str): The name of the collection to query.
        llm (ChatOpenAI): An instance of the ChatOpenAI object.
        embedding_model (OpenAIEmbeddings): An instance of the OpenAIEmbeddings object.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        str: The generated question.
    """
    retriever = setup_chroma_retriever(collection_name=collection,
                                       embedding_model=embedding_model,
                                       verbose=verbose)
    st.session_state.retriever = retriever
    generated_question = generate_question(llm=llm,
                                           retriever=st.session_state.retriever,
                                           previous_questions=st.session_state.previous_questions,
                                           verbose=verbose)
    generated_answer = generate_answer(llm=llm,
                                       retriever=st.session_state.retriever,
                                       question=generated_question,
                                       verbose=verbose)

    st.session_state.current_question = generated_question
    st.session_state.generated_answer = generated_answer
    st.session_state.previous_questions.append(generated_question)
    return generated_question

def delete_file(filename, docs_path, verbose=False):
    """
    Deletes the selected file and its associated collection.

    Args:
        filename (str): The name of the file to delete.
        docs_path (str): The path to the files directory.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Raises:
        ValueError: If the file to delete is not the current session file.
    """
    try:
        if st.session_state.filename != filename:
            raise ValueError(f"Cannot delete file \"{filename}\" because it's not the current session file.")

        st.session_state.chat_history.clear()
        st.session_state.filename = None
        os.remove(os.path.join(docs_path, filename))
        st.session_state.uploaded_files.remove(filename)
        persistent_client = PersistentClient(path=DB_PATH)
        persistent_client.delete_collection(st.session_state.collection_name)
        if verbose:
            print(f"Collection {st.session_state.collection_name} deleted.")  
    except ValueError as e:
        st.error("Error occured: " + str(e))    
    
def clear_chat_history(filename):
    """Clears the chat history and previous questions for the selected file."""
    if st.session_state.filename == filename:
        st.session_state.chat_history.clear()
        st.session_state.previous_questions.clear()
        check_collection_exists(st.session_state.collection_name)

def get_uploaded_file_names(docs_path):
    """Returns a list of the names of the uploaded files."""
    return os.listdir(docs_path)

def response_generator(response):
    """Generates a response with a delay for a more natural chat experience."""
    for word in response.split():
        yield word + " "
        time.sleep(0.05)