import os
import streamlit as st
import re
from langchain_community.document_loaders import OnlinePDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

if "document_url" not in st.session_state:
    st.session_state["document_url"] = ""


def load_pdf(url=st.session_state["document_url"]):
    loader = OnlinePDFLoader(url)
    page = loader.load_and_split()

    print(page)
    return page


def create_embeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store


def load_llm_model():
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0, "max_length": 2048},
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def ask_questions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response


def create_vector_store():
    loaded_docs = load_pdf()
    st.session_state["vector_store"] = create_embeddings(loaded_docs)
    st.write("Vector Store Created")


def extract_helpful_answer(response):
    # Define a regular expression pattern to search for "Helpful Answer:" followed by any text
    # Adjust the pattern as necessary, depending on the format of your responses
    pattern = r"Helpful Answer:\s*(.*)"

    # Search for the pattern in the response
    match = re.search(pattern, response, re.IGNORECASE)

    # If a match is found, return the captured group (text after "Helpful Answer:")
    if match:
        return match.group(1).strip()  # .strip() removes leading/trailing whitespace
    else:
        return "Helpful answer not found in the response."


# Example usage in your run_ask_questions function
def run_ask_questions():
    if (
        st.session_state["vector_store"] is not None
        and st.session_state["user_question"]
    ):
        chain = load_llm_model()
        full_response = ask_questions(
            st.session_state["vector_store"], chain, st.session_state["user_question"]
        )
        helpful_answer = extract_helpful_answer(full_response)
        return helpful_answer
    else:
        st.error(
            "Vector store is not initialized or question is not provided. Please initialize and enter a question."
        )


# Streamlit UI
st.title("Private LLM")

st.session_state["document_url"] = st.text_input("Enter your document link:")

if st.button("Initialize Vector Store"):
    with st.spinner("Initializing Vector Store..."):
        create_vector_store()
        st.success("Vector store initialized successfully!")

# User inputs the question here
st.session_state["user_question"] = st.text_input("Enter your question:")

if st.button("Ask Questions"):
    with st.spinner("Asking Questions..."):
        response = run_ask_questions()
        if response:
            st.write(response)
