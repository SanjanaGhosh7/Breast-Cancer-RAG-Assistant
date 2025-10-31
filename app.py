# ============================================
# 🌸 Breast Cancer RAG Assistant
# ============================================

# Imports for the Breast Cancer RAG Assistant
import os
from dotenv import load_dotenv
import requests
import streamlit as st

# LangChain community integrations
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LangChain core + text + prompts
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Groq LLM (for generation)
from langchain_groq import ChatGroq

# Load the environment variable
load_dotenv()  #Save your GROQ_API_KEY in .env file


# ============================================
# 🧩 STEP 1: Fetch Research Papers from PubMed
# ============================================
def fetch_pubmed_data(query, max_results=10):
    """Fetch PubMed abstracts related to the given query"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Search for PubMed IDs
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    search_results = requests.get(search_url).json()
    ids = search_results.get("esearchresult", {}).get("idlist", [])

    # Fetch abstracts for those IDs
    abstracts = []
    for pmid in ids:
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(fetch_url)
        if response.ok:
            abstracts.append(response.text)
    return abstracts


# ============================================
# 🧬 STEP 2: Convert Abstracts into Documents
# ============================================
def prepare_documents(abstracts):
    """Convert text abstracts into LangChain Document objects"""
    docs = []
    for abs_text in abstracts:
        docs.append(Document(page_content=abs_text))
    return docs


# ============================================
# 🧠 STEP 3: Create Vector DB (Chroma)
# ============================================
def create_vector_db(docs):
    """Split text, embed it, and store in Chroma"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Use a small, beginner-friendly sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb


# ============================================
# 🤖 STEP 4: Build the RAG Chain
# ============================================
def build_rag_chain(vectordb):
    """Set up retrieval + generation pipeline using Groq LLM"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature = 0
    )

    # Prompt Template for LLM
    template = """You are an AI research assistant specialized in cancer biology.
Use the provided context from PubMed papers to answer the question clearly and simply.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def rag_answer(question):
        docs = docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        return response.content

    return rag_answer


# ============================================
# 💻 STEP 5: Streamlit UI
# ============================================
def main():
    st.title("🌸 Breast Cancer RAG Assistant")
    st.write("Ask me about **phytochemicals targeting breast cancer biomarkers.**")

    if "vectordb" not in st.session_state:
        st.session_state["vectordb"] = None

    # Question input
    user_q = st.text_input("🔎 Ask a question:")

    # Fetch & build database
    if st.button("📚 Fetch Research Papers & Get Answer"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Fetching data from PubMed..."):
            abstracts = fetch_pubmed_data("breast cancer biomarkers phytochemicals", max_results=10)
            if not abstracts:
                st.error("No abstracts found. Try again later.")
                return

            docs = prepare_documents(abstracts)
            vectordb = create_vector_db(docs)
            st.session_state["vectordb"] = vectordb



        if user_q and st.session_state["vectordb"]:
            rag_fn = build_rag_chain(st.session_state["vectordb"])
            answer = rag_fn(user_q)

        #Show answer line-by-line
        st.write("### 🧠 Answer:")
        st.write(answer)

    st.markdown("---")
    st.caption("Built using PubMed + Chroma + Groq LLM")


# Run the app
if __name__ == "__main__":
    main()
