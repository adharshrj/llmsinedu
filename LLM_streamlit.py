import os
import streamlit as st
import re
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import pinecone


load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

PINECONE_API = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

if "document_url" not in st.session_state:
    st.session_state["document_url"] = ""

index_name = "quickstart"
embedder = HuggingFaceEmbeddings()
pc = pinecone.Pinecone(api_key=PINECONE_API, environment=PINECONE_ENV)
st.session_state["vector_store"] = Pinecone.from_existing_index(index_name, embedder)


def load_pdf(url=st.session_state["document_url"]):
    loader = OnlinePDFLoader(url)
    page = loader.load()

    print(page)
    return page


def split_document(loaded_docs):
    try:
        # Splitting documents into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs
    except Exception as e:
        # Handle the exception or log the error
        print(f"Error splitting document: {str(e)}")
        return None


def create_embeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    print(f"PINCONE_API: {PINECONE_API}, PINCONE_ENV: {PINECONE_ENV}")
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, metric="cosine", dimension=768)

    vector_store = Pinecone.from_documents(
        chunked_docs, embedder, index_name="quickstart"
    )
    return vector_store


# mistralai/Mistral-7B-Instruct-v0.1
# mistralai/Mistral-7B-Instruct-v0.2
# mistralai/Mixtral-8x7B-Instruct-v0.1
# openchat/openchat-3.5-0106


def load_llm_model():
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.7, "max_length": 20000, "max_new_tokens": 20000},
    )
    print(llm)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def ask_questions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response


def create_vector_store():
    loaded_docs = split_document(load_pdf())
    st.session_state["vector_store"] = create_embeddings(loaded_docs)
    st.write("Vector Store Created")


def extract_helpful_answer(response):
    # Define a regular expression pattern to search for "Answer:" followed by any text, including multiline
    pattern = r"Answer:\s*(.*)"

    # Search for the pattern in the response, with DOTALL flag to capture multiline text
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

    # If a match is found, return the captured group (text after "Answer:")
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
        # return full_response
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

if st.button("Ask Question"):
    with st.spinner("Asking Questions..."):
        response = run_ask_questions()
        if response:
            st.write(response)
