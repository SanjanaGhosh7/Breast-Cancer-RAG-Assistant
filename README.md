🌸 **Breast Cancer RAG Assistant**

An AI-powered assistant that retrieves and summarizes scientific literature on phytochemicals targeting breast cancer biomarkers.

🧠 **Overview**

The Breast Cancer RAG Assistant is a beginner-friendly GenAI project that demonstrates the Retrieval-Augmented Generation (RAG) workflow in a real-world biomedical use case.

It uses the PubMed API to fetch research papers, stores and retrieves embeddings via ChromaDB, and uses a Groq-hosted LLaMA 3 model to generate human-like, research-informed answers — all wrapped in an interactive Streamlit web app.

🚀 **Features**

🧬 Fetches real research papers from PubMed using NCBI’s E-utilities API

🔍 Embeds and retrieves relevant information using LangChain + ChromaDB

💬 Generates concise, scientific answers with Groq LLM

🌿 Focuses on phytochemicals and breast cancer biomarkers

💻 Interactive and easy-to-use Streamlit UI

☁️ Fully deployed on Streamlit Cloud

🧩 **Tech Stack**

| Category            | Tools / Libraries         |
| ------------------- | ------------------------- |
| **Language**        | Python                    |
| **Frameworks**      | Streamlit, LangChain      |
| **LLM Provider**    | Groq (LLaMA 3)            |
| **Vector Database** | ChromaDB                  |
| **Embeddings**      | HuggingFace               |
| **APIs**            | PubMed (NCBI E-utilities) |
| **Deployment**      | Streamlit Cloud           |
| **Version Control** | Git/GitHub                |

⚙️ **How It Works**
🧮 Step 1: Retrieve Data

The app connects to the PubMed API to fetch abstracts on “breast cancer biomarkers phytochemicals”.

🧱 Step 2: Process & Store

Fetched text is split into smaller chunks using LangChain’s text splitter, then embedded using HuggingFace sentence transformers.

💾 Step 3: Store in Vector Database

Embeddings are stored in ChromaDB, which enables semantic similarity-based retrieval.

🧠 Step 4: Generate Answers (RAG Pipeline)

When the user asks a question:

Relevant papers are retrieved from Chroma.

The selected context is passed to a Groq-hosted LLaMA 3 model.

The model generates a context-aware, concise response.

💻 Step 5: User Interface

Built using Streamlit, the app allows users to:

Fetch new papers

Ask custom questions

View AI-generated answers 

🧰 **Installation**
1️⃣ Clone the repository
git clone https://github.com/SanjanaGhosh7/breast-cancer-rag-assistant.git
cd breast-cancer-rag-assistant

2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
.venv\Scripts\activate      # on Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your Groq API key

Create a .env file in the project root and paste this:

GROQ_API_KEY=your_api_key_here

5️⃣ Run the app
streamlit run app.py

Then open http://localhost:8501
 in your browser.

🧬 **Example Query**

**Question:**
“What are the top 3 phytochemicals targeting HER2 pathway in breast cancer?”

**Sample Output:**
“Based on the provided context from the PubMed paper "Breast Tumor Microenvironment: Emerging Target of Therapeutic Phytochemicals" by Malla R. R. et al. (2020), the top 3 phytochemicals targeting the HER2 pathway in breast cancer are:

**Curcumin:** Curcumin, a polyphenol derived from turmeric, has been shown to inhibit the HER2 pathway by suppressing the expression of HER2 protein and its downstream signaling molecules.
**Quercetin:** Quercetin, a flavonoid found in various fruits and vegetables, has been reported to inhibit the HER2 pathway by blocking the activation of HER2 receptor and its downstream signaling molecules.
**Gingerol:** Gingerol, a bioactive compound derived from ginger, has been shown to inhibit the HER2 pathway by suppressing the expression of HER2 protein and its downstream signaling molecules.
These phytochemicals have been identified as potential therapeutic agents for targeting the HER2 pathway in breast cancer, and further research is needed to explore their efficacy and safety in clinical settings.”

📂 **Project Structure**

📁 Breast-Cancer-RAG-Assistant
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── .env                    # API keys (not uploaded)
├── chroma_db/              # Auto-generated vector store
├── rag.png                 # Screenshot of the App
└── README.md               # Project documentation

📚 **Concepts Learned**

🧠 RAG (Retrieval-Augmented Generation) architecture

🔡 Sentence Embeddings using HuggingFace Transformers

🗃️ Vector databases and semantic search (ChromaDB)

🧩 LangChain integration (Document loading, splitting, retrieval)

🧬 PubMed API data fetching and preprocessing

💬 LLM prompt engineering using Groq

🌐 Streamlit web app creation and deployment

🌍 **Deployment**

Deployed live on Streamlit Cloud:
👉 https://breast-cancer-rag-assistant-gvvfh52rsefmbnsshgtf68.streamlit.app/

🧑‍💻 **Author**

Sanjana Ghosh
Bioinformatics Postgraduate | Exploring AI?GenAI in Biopharma
🔗 LinkedIn: https://www.linkedin.com/in/sanjana-ghosh-2a5b7c11d/ 
• GitHub: https://github.com/SanjanaGhosh7
• Medium: https://medium.com/@sanjanaghosh25 

🧭 **Future Improvements**

Integrate FAISS or Pinecone for scalable vector search

Extend to other cancers and molecular targets

Add PDF uploading for user-provided literature

💖 **Acknowledgements**

LangChain for RAG framework

Groq for ultra-fast LLM inference

HuggingFace for embeddings

PubMed / NCBI for biomedical data access

Streamlit for easy app deployment

📜 **License**

This project is open-source and available under the MIT License.
