import os
import getpass
import streamlit as st
from langchain import hub
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.vectorstores import InMemoryVectorStore

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangChain API key: ")

# Check for Hugging Face and Mistral AI API keys
if not os.environ.get("HF_API_KEY"):
    os.environ["HF_API_KEY"] = getpass.getpass("Enter API key for Hugging Face: ")

if not os.environ.get("MISTRALAI_API_KEY"):
    os.environ["MISTRALAI_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")

# Initialize embeddings and LLMs
embeddings = MistralAIEmbeddings(model="mistral-embed")
llm = ChatMistralAI(model="mistral-large-latest")

# Create vector store
vector_store = InMemoryVectorStore(embeddings)

# Load CSV document
file_path = "path_to_your_screentime_analysis.csv"  # Adjust this path
loader = CSVLoader(file_path=file_path)
docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks in vector store
vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the application steps for retrieval and generation
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile the application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit UI to interact with the model
st.title("Screentime Analysis QA")

# Input field for the user's question
question = st.text_input("Ask a question about screentime data:")

# Display the answer when a question is asked
if question:
    response = graph.invoke({"question": question})
    st.write("Answer:", response["answer"])
