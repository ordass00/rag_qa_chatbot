import os
import streamlit as st
from app.utils import create_docs_directory, prepare_collection_name
from app.chatbot_logic import init_llm, get_uploaded_file_names, clear_chat_history, delete_file, select_file, response_generator
from app.initialize_state_variables import init_session_state_variables
from app.vector_logic import init_chromadb, check_collection_exists
from app.chain_logic import generate_question, generate_answer, evaluate_qa, evaluate_qa_with_cosine_similarity
from app.constants import DOCS_PATH, DB_PATH
from langchain_core.messages import HumanMessage, AIMessage

# Set verbose to True/False to display/hide additional information in the console
verbose = True

st.set_page_config(page_title="🤖💬 QA Bot")
st.sidebar.title("🤖💬 OpenAI QA Chatbot")

llm, embedding_model = init_llm()
init_session_state_variables()
create_docs_directory(DOCS_PATH)
create_docs_directory(DB_PATH)

# Upload button for docx files in sidebar
uploaded_file = st.sidebar.file_uploader(label="Upload a docx file",
                                         type="docx")

# Check if a file is uploaded
if uploaded_file is not None:
    if verbose:
        print("Uploading file...")
    filename = uploaded_file.name
    st.session_state.collection_name = prepare_collection_name(filename)
    st.session_state.file_in_upload_form = True
    file_path = os.path.join(DOCS_PATH, filename)

    if verbose:
        print("Filename: ", filename)
        print("Collection name: ", st.session_state.collection_name)
        print("File path: ", file_path)

    # Check if a file with the same name already exists
    if os.path.isfile(file_path):
        st.error(f"A file with file name \"{filename}\" already exists. Please upload a different file.")
        st.stop()

    # Save the uploaded file to the docs folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.session_state.file_uploaded = True
    st.session_state.uploaded_files.append(filename)

    # Reset session state chat history and filename after uploading a new file
    st.session_state.chat_history.clear()
    st.session_state.previous_questions.clear()
    st.session_state.filename = None
    uploaded_file = None
    st.success("File uploaded successfully!")

    # Check if the collection for the current file already exists in the database
    is_collection = check_collection_exists(
        collection_name=st.session_state.collection_name)
    if verbose:
        print("Is collection: ", is_collection)

    # If the collection doesn't exist, create ChromaDB vectorstore for it
    if is_collection["status"] != "success":
        init_chromadb(collection_name=st.session_state.collection_name,
                      embedding_model=embedding_model,
                      file_path=file_path,
                      verbose=verbose)

    with st.chat_message("assistant"):
        st.write('File uploaded. Click "X" to remove it from the upload & click "Chat" to start the conversation.')

# Inject custom CSS for the app
with open("app/app.css") as f:
    css_style = f"<style>{f.read()}</style>"
st.sidebar.markdown(css_style, unsafe_allow_html=True)

# Display file interaction buttons only if there are files available
files = get_uploaded_file_names(DOCS_PATH)
if files:
    st.session_state.uploaded_files = files
    selected_file = st.sidebar.selectbox(label="Select a file", options=st.session_state.uploaded_files, disabled=st.session_state.file_in_upload_form)
    st.session_state.collection_name = prepare_collection_name(selected_file)
    st.session_state.filename = selected_file
    if verbose:
        print("Current collection name: ", st.session_state.collection_name)
        print("Current filename: ", st.session_state.filename)

    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        st.button(label="Chat", on_click=select_file, args=(selected_file, llm, embedding_model, verbose), disabled=st.session_state.file_in_upload_form)
    with col2:
        st.button(label="Clear Chat", on_click=clear_chat_history, args=(selected_file, ), disabled=st.session_state.file_in_upload_form)
    with col3:
        st.button(label="Delete", on_click=delete_file, args=(selected_file, DOCS_PATH, verbose), disabled=st.session_state.file_in_upload_form)

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if st.session_state.filename:
    user_input = st.chat_input(disabled=st.session_state.file_in_upload_form)
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append(HumanMessage(user_input))

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                # Evaluate user input and output the correct answer
                evaluation = evaluate_qa_with_cosine_similarity(
                    embedding_model=embedding_model,
                    correct_answer=st.session_state.generated_answer,
                    user_answer=user_input,
                    verbose=verbose)

                st.write_stream(response_generator(evaluation))
        
        with st.chat_message("assistant"):
            with st.spinner("Asking the next question..."):
                # Generate the next question
                question = generate_question(llm=llm,
                                             retriever=st.session_state.retriever,
                                             previous_questions=st.session_state.previous_questions,
                                             verbose=verbose)
                # Generate the answer to the new question
                answer = generate_answer(llm=llm,
                                         retriever=st.session_state.retriever,
                                         question=question,
                                         verbose=verbose)

                st.session_state.current_question = question
                st.session_state.generated_answer = answer
                st.session_state.previous_questions.append(question)

                st.write_stream(response_generator(question))

        st.session_state.chat_history.append(AIMessage(evaluation))
        st.session_state.chat_history.append(AIMessage(question))

if uploaded_file is None:
    st.session_state.file_in_upload_form = False