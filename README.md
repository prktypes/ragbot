# RAGBot (RAG-powered | Streamlit + Ollama + Qdrant)

A secure, production-ready **RAG (Retrieval-Augmented Generation)** chatbot application built with **Streamlit**, featuring:

🚀 Local LLMs via [Ollama](https://ollama.com)  
🔐 Authentication with [Auth0](https://auth0.com)  
🧠 Context-aware Q&A from organisation PDFs  
📄 Dynamic PDF upload + chat  
⚡ Fast vector search using [Qdrant](https://qdrant.tech)  
🎨 Beautiful, dark-themed UI with full chat history and user isolation

---

## ✨ Demo Screenshot

<img width="1400" height="975" alt="1" src="https://github.com/user-attachments/assets/9a8c27be-cd91-40e9-92ac-e3de6712b11b" />
<img width="1415" height="978" alt="Screenshot 2025-07-18 021601" src="https://github.com/user-attachments/assets/5b21dec6-8f49-4517-9224-990c1ec5d1a0" />
<img width="1407" height="975" alt="3" src="https://github.com/user-attachments/assets/cb8eff1a-67ed-42c3-8c79-cc8469a4c0a8" />


---

## 📌 Features

- ✅ **User Authentication (OAuth2)** via Auth0 (Google login)
- ✅ **Chat with Your Documents**: PDF upload + RAG pipeline
- ✅ **LLM-powered Answers** using Ollama (Mistral model)
- ✅ **Memory-enabled Conversations**
- ✅ **Persistent Chat History per User**
- ✅ **Secure File Handling**
- ✅ **Elegant Dark UI** (custom CSS)
- ✅ **Search-aware vector store (MMR)** using Qdrant
- ✅ **Download, rename, or delete chats**
- ✅ **Split & Embed PDFs** using `RecursiveCharacterTextSplitter` + `HuggingFaceEmbeddings`

---

## 🧠 Architecture

```text
[PDF Upload] ---> [PyPDFLoader]
                          |
                [Text Chunking]
                          |
            [HuggingFace Embeddings]
                          |
        --> [Qdrant Vector DB] <---
        |                          |
    [User Prompt]         [Retriever (MMR)]
            |                      |
         [Ollama (Mistral)] ←-----|
                |
       [Conversational Chain]
                |
         [Streamlit UI]

```

 🔐 Authentication

- 🔐 OAuth 2.0 via **Auth0**
- 🧑 Users can **sign up / log in with Google**
- 💾 Session state persists via **Streamlit**
- 🧍 Each user has **isolated chat history**

---

## 🧾 Requirements

| Component     | Usage                          |
|---------------|--------------------------------|
| Python 3.10+  | Backend logic                  |
| Streamlit     | Frontend & interactivity       |
| Ollama        | Local LLM engine (Mistral)     |
| Qdrant        | Vector database for retrieval  |
| HuggingFace   | Text embeddings                |
| Auth0         | Secure authentication          |

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the repository

git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Set up .env file
Create a .env file in the root with your Auth0 credentials:

AUTH0_DOMAIN=your-auth0-domain
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-secret

4️⃣ Start Qdrant (via Docker)

docker run -d -p 6333:6333 qdrant/qdrant
5️⃣ Run Ollama (install & pull model)

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server & pull Mistral model

ollama serve &
ollama pull mistral
6️⃣ Start the Streamlit app

streamlit run app.py --server.address 0.0.0.0 --server.port 8501
