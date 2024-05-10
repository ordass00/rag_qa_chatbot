<h1 align="center">RAG Question-Answer Chatbot</h1>

<div style="background-color:white">
  <div align="center">
    <img src="./Imgs/techfak_logo.svg" width="700" height="250">
    <hr>
    <h2 style="color:black">Applications of LLMs - Interactive Communication System<h2>
    <h3 style="color:black">Oliver Dassinger<h3>
    <img src="./Imgs/pr_lab.png" width="200">
  </div>
  <hr>
</div>



## Abstract
The recent advancements in Large Language Models (LLMs) have significantly improved capabilities in various domains, such as content creation, translation, code generation, customer support, and education. These improvements allowed the 
creation of sophisticated interactive applications like chatbots. Despite this progress, they still face limitations as their training relies on large static datasets which quickly become outdated as well as the inability to access private data. This project develops a document-based question-answering chatbot using the LangChain framework to utilize the OpenAI API for deploying an LLM. It makes use of the Retrieval-Augmented Generation (RAG) technique to dynamically generate questions based on the document embeddings stored through ChromaDB. By reversing the typical question-answer interaction format, the system proactively questions the user about the content of the provided document. The user response is then evaluated by using the cosine similarity to determine the accuracy against the correct, model-generated answer. This approach successfully demonstrates the advantages of using RAG for interactive question-answering, significantly enhancing the adaptability and interactivity of chatbots.


### Structure

```

+-- .streamlit
|   +-- config.toml                                              # The Streamlit server settings
|
+-- Imgs
|   +-- Images used for the GitHub repo
|
+-- Report
|   +-- Images                        
|   |    +-- chatbot_interface.png
|   |    +-- overview_system_architecture.png
|   +-- Applications_of_LLMs.pdf                                  # The final project report
|   +-- references.bib
|   +-- report.tex
|
+-- app
|   +-- app.css                                                   # Custom CSS for Streamlit App - Sidebar and Button Styling
|   +-- chain_logic.py                                            # The chain logic for Q&A generation and evaluation
|   +-- chatbot_logic.py                                          # The chatbot logic, initialization & interaction handling
|   +-- constants.py                                              # The constants for the application
|   +-- initialize_state_variables.py                             # The initialization of the session state variables
|   +-- utils.py                                                  # The utility functions for handling document directories & text extraction
|   +-- vector_logic.py                                           # The vector storage & retrieval of document embeddings
|
+-- .gitignore
+-- README.md
+-- main.py                                                       # The core file & entry point for the Streamlit App     
+-- requirements.txt                    

```
## Links to Ressources

- Final Report as [PDF](https://github.com/ordass00/rag_qa_chatbot/blob/main/Report/Application_of_LLMs.pdf)

## Ressources
- LangChain: https://www.langchain.com/langchain
- LangChain OpenAI API integration: https://python.langchain.com/v0.1/docs/integrations/platforms/openai/
- ChromaDB: https://www.trychroma.com/
- Streamlit: https://streamlit.io/
- Pattern Recognition Lecture Notes: https://lme.tf.fau.de/category/lecture-notes/lecture-notes-pr/
- FAU Pattern Recognition Lab: https://lme.tf.fau.de/

### Prerequisites

```
The dependencies to this project are stored in the file:
   - requirements.txt

I used python version 3.12.2
```

## How to Use

1. Upon running the Streamlit command, you need to set the OpenAI API key in your environment variables.
2. Type following command into your console while being in the right folder. This starts the Streamlit application.

```
  streamlit run main.py
```

## Author

* **Oliver Dassinger**
