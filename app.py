from dotenv import load_dotenv
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
import os
import json
from datetime import datetime, timedelta
import streamlit as st
import jwt
import requests
import os
import re
from urllib.parse import urlencode
import tempfile
import shutil
from langchain.docstore.document import Document

load_dotenv()

def login_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
login_css("login.css")

def apply_chatbot_theme(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_chatbot_theme("chatbot.css")


# -------------------- Auth0 Setup --------------------

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8501"
AUTH0_BASE_URL = f"https://{AUTH0_DOMAIN}"

def login_url():
    query = {
        "client_id": AUTH0_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid profile email",
        "audience": f"https://{AUTH0_DOMAIN}/userinfo"
    }
    return f"{AUTH0_BASE_URL}/authorize?{urlencode(query)}"

def exchange_code_for_token(code):
    token_url = f"{AUTH0_BASE_URL}/oauth/token"
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    body = {
        "grant_type": "authorization_code",
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    response = requests.post(token_url, headers=headers, data=body)
    return response.json()

def parse_code():
    params = st.query_params
    # Fix: Handle both old and new streamlit query params format
    if hasattr(params, 'get'):
        return params.get("code")
    else:
        return params.get("code", [None])[0] if "code" in params else None

def get_user_info(token):
    resp = requests.get(f"{AUTH0_BASE_URL}/userinfo", headers={"Authorization": f"Bearer {token}"})
    return resp.json()

def logout_user():
    """Clear session state and redirect to logout URL"""
    # Clear user-specific session state but keep the current user info for cleanup
    current_user_id = get_user_id()
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Redirect to Auth0 logout
    logout_url = f"{AUTH0_BASE_URL}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo={REDIRECT_URI}"
    st.markdown(f'<meta http-equiv="refresh" content="0; url={logout_url}">', unsafe_allow_html=True)
    st.stop()

def initialize_user_session():
    """Initialize user-specific session state"""
    user_id = get_user_id()
    
    # Initialize chat history if not exists or if user changed
    if ("chat_history" not in st.session_state or 
        "current_user_id" not in st.session_state or 
        st.session_state.get("current_user_id") != user_id):
        
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.current_user_id = user_id
        st.session_state.chat_title = None
        st.session_state.vectordb = None
        st.session_state.qa_chain = None
        st.session_state.uploaded_filename = None

# -------------------- Login Logic --------------------

def signup_url():
    query = {
        "client_id": AUTH0_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid profile email",
        "audience": f"https://{AUTH0_DOMAIN}/userinfo",
        "screen_hint": "signup"
    }
    return f"{AUTH0_BASE_URL}/authorize?{urlencode(query)}"


if "user" not in st.session_state:
    code = parse_code()
    if code:
        try:
            tokens = exchange_code_for_token(code)
            access_token = tokens.get("access_token")
            if access_token:
                user_info = get_user_info(access_token)
                st.session_state["user"] = user_info
                st.session_state["logged_in"] = True
                # Clear the code from URL
                st.query_params.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            st.session_state["logged_in"] = False
    else:
        st.session_state["logged_in"] = False
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h1 style="font-size: 2.5rem; font-weight: bold; color: white;">Welcome</h1>
            </div>
            <div class="login-container">
                <h2>Login to your account</h2>
                <p>Click below to login with Google</p>
                <a href="{login_url()}">
                    <button class="login-button google">Login with Google</button>
                </a>
                <div class="login-footer">
                    Don't have an account?
                    <a href="{signup_url()}" style="color: #0ea5e9; text-decoration: underline;">Sign up</a>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

# -------------------- Setup --------------------

LOG_DIR = "chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_user_id():
    """Get the current user's unique ID"""
    if "user" in st.session_state and st.session_state["user"]:
        return st.session_state["user"].get("sub", "anonymous")
    return "anonymous"

def get_user_log_dir():
    """Get user-specific log directory"""
    user_id = get_user_id()
    # Create a safe directory name from user ID
    safe_user_id = re.sub(r'[^\w\-_]', '_', user_id)
    user_dir = os.path.join(LOG_DIR, safe_user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_today_date():
    return datetime.now().strftime("%Y-%m-%d")

def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[\s]+", "_", text)

def generate_chat_filename(title):
    date = get_today_date()
    slug = slugify(title)[:30]
    return f"{date}_{slug}.json"

def save_full_chat(title, chat):
    filename = generate_chat_filename(title)
    user_dir = get_user_log_dir()
    filepath = os.path.join(user_dir, filename)
    with open(filepath, "w") as f:
        json.dump({
            "title": title,
            "date": get_today_date(),
            "timestamp": get_timestamp(),
            "messages": chat,
            "user_id": get_user_id(),
            "uploaded_filename": st.session_state.get("uploaded_filename", None)
        }, f)

def load_chat_logs():
    user_dir = get_user_log_dir()
    today = []
    previous = []
    
    if not os.path.exists(user_dir):
        return today, previous
        
    for file in os.listdir(user_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(user_dir, file)) as f:
                    data = json.load(f)
                    # Skip files in old format (list instead of dict)
                    if isinstance(data, dict) and "date" in data and "title" in data and "messages" in data:
                        # Additional security check: ensure chat belongs to current user
                        if data.get("user_id") == get_user_id():
                            if data["date"] == get_today_date():
                                today.append(data)
                            else:
                                previous.append(data)
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return today, previous

def delete_all_chats():
    user_dir = get_user_log_dir()
    if not os.path.exists(user_dir):
        return
        
    for file in os.listdir(user_dir):
        if file.endswith(".json"):
            try:
                # Additional security check
                with open(os.path.join(user_dir, file), "r") as f:
                    data = json.load(f)
                    if data.get("user_id") == get_user_id():
                        os.remove(os.path.join(user_dir, file))
            except Exception as e:
                print(f"Error deleting {file}: {e}")

def delete_today_chats():
    user_dir = get_user_log_dir()
    if not os.path.exists(user_dir):
        return
        
    for file in os.listdir(user_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(user_dir, file), "r") as f:
                    data = json.load(f)
                    if data.get("date") == get_today_date() and data.get("user_id") == get_user_id():
                        os.remove(os.path.join(user_dir, file))
            except Exception as e:
                print(f"Error deleting today's chat {file}: {e}")

def delete_old_chats():
    user_dir = get_user_log_dir()
    if not os.path.exists(user_dir):
        return
        
    for file in os.listdir(user_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(user_dir, file), "r") as f:
                    data = json.load(f)
                    if data.get("user_id") == get_user_id():
                        chat_date = datetime.strptime(data.get("date"), "%Y-%m-%d")
                        if datetime.now() - chat_date > timedelta(days=7):
                            os.remove(os.path.join(user_dir, file))
            except Exception as e:
                print(f"Error deleting old chat {file}: {e}")

def download_chat_logs():
    user_dir = get_user_log_dir()
    if not os.path.exists(user_dir):
        return
        
    for file in os.listdir(user_dir):
        if file.endswith(".json"):
            filepath = os.path.join(user_dir, file)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    # Security check
                    if data.get("user_id") == get_user_id():
                        chat_data = json.dumps(data, indent=2)
                        st.download_button(
                            label=f"📥 Download {file}",
                            data=chat_data,
                            file_name=file,
                            mime="application/json"
                        )
            except Exception as e:
                print(f"Error reading {file}: {e}")

# -------------------- PDF Processing Functions --------------------

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF file and create vector store"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store using Chroma (in-memory)
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectordb, len(text_chunks)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0

def create_qa_chain(vectordb, llm):
    """Create conversational retrieval chain"""
    retriever = vectordb.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory
    )
    
    return qa_chain

# -------------------- Load Model --------------------

@st.cache_resource
def load_llm():
    return Ollama(model="mistral")

llm = load_llm()

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Company Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
            font-family: apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";!important;
        }
        .stTextInput, .stTextArea, .stButton > button {
            background-color: #2D2D2D !important;
            color: white !important;
        }
        .stChatMessage {
            background-color: #2D2D2D;
        }
        .user-info {
            background-color: #2D2D2D;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid #444;
        }
        .logout-btn {
            background-color: #dc3545 !important;
            color: white !important;
            border: none !important;
            margin-bottom: 10px;
        }
        .logout-btn:hover {
            background-color: #c82333 !important;
        }
        .upload-info {
            background-color: #2D2D2D;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid #444;
        }
        .upload-success {
            background-color: #155724;
            color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1 class="title">RAG Chatbot</h1>
    <p class="subtitle">Upload a PDF and ask questions about its content.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar: User Info and Logout --------------------
with st.sidebar:
    # User info and logout at the top
    if st.session_state.get("user") and st.session_state.get("logged_in"):
        user_info = st.session_state["user"]
        user_name = user_info.get("name", "User")
        user_email = user_info.get("email", "No email")
        user_picture = user_info.get("picture", "")
        
        # Display user info
        st.markdown(
            f"""
            <div class="user-info">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    {f'<img src="{user_picture}" width="40" height="40" style="border-radius: 50%; margin-right: 10px;">' if user_picture else '👤'}
                    <div>
                        <div style="font-weight: bold; color: white;">{user_name}</div>
                        <div style="font-size: 12px; color: #ccc;">{user_email}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Logout button
        if st.button("Logout", key="logout_btn", use_container_width=True, help="Click to logout"):
            logout_user()
    
    st.markdown("---")

# -------------------- PDF Upload Section --------------------
with st.sidebar:
    st.subheader("Upload Your PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file to create a knowledge base for the chatbot"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file or same file
        if (st.session_state.get("uploaded_filename") != uploaded_file.name or 
            st.session_state.get("vectordb") is None):
            
            with st.spinner("Processing PDF..."):
                vectordb, num_chunks = process_uploaded_pdf(uploaded_file)
                
                if vectordb is not None:
                    st.session_state.vectordb = vectordb
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.qa_chain = create_qa_chain(vectordb, llm)
                    
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"📊 Created {num_chunks} text chunks from your PDF")
                else:
                    st.error("Failed to process PDF. Please try again.")
    
    # Show current uploaded file info
    if st.session_state.get("uploaded_filename"):
        st.markdown(
            f"""
            <div class="upload-info">
                <div style="font-size: 14px; color: #28a745;">
                    Current PDF: {st.session_state.uploaded_filename}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

# -------------------- Sidebar: Chat History --------------------
def styled_sidebar_button(label, callback):
    with st.sidebar:
        if st.button(label, key=label):
            callback()

with st.sidebar:
    st.title("Chat History")

today_chats, past_chats = load_chat_logs()

st.sidebar.subheader("Today")
for chat in today_chats:
    with st.sidebar.expander(chat["title"], expanded=False):
        # Show associated PDF filename if available
        if chat.get("uploaded_filename"):
            st.markdown(f"📄 *{chat['uploaded_filename']}*")
        
        if st.button("Open Chat", key=f"open_{chat['timestamp']}"):
            st.session_state.chat_history = chat["messages"]
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            for turn in chat["messages"]:
                st.session_state.memory.chat_memory.add_user_message(turn["user"])
                st.session_state.memory.chat_memory.add_ai_message(turn["ai"])
            st.session_state.chat_title = chat["title"]
            st.rerun()

        new_title = st.text_input("Rename Chat", value=chat["title"], key=f"rename_{chat['timestamp']}")
        if new_title != chat["title"]:
            # Rename the file and update title inside JSON
            old_filename = generate_chat_filename(chat["title"])
            new_filename = generate_chat_filename(new_title)
            user_dir = get_user_log_dir()
            old_path = os.path.join(user_dir, old_filename)
            new_path = os.path.join(user_dir, new_filename)

            if os.path.exists(old_path):
                with open(old_path, "r") as f:
                    data = json.load(f)
                # Security check
                if data.get("user_id") == get_user_id():
                    data["title"] = new_title
                    with open(new_path, "w") as f:
                        json.dump(data, f, indent=2)
                    os.remove(old_path)
                    st.success("Renamed successfully!")
                    st.rerun()


with st.sidebar.expander("Previous 30 Days", expanded=False):
    for chat in past_chats:
        # Show associated PDF filename if available
        if chat.get("uploaded_filename"):
            st.markdown(f"📄 *{chat['uploaded_filename']}*")
            
        if st.button("Open Chat", key=f"open_{chat['timestamp']}"):
            st.session_state.chat_history = chat["messages"]
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            for turn in chat["messages"]:
                st.session_state.memory.chat_memory.add_user_message(turn["user"])
                st.session_state.memory.chat_memory.add_ai_message(turn["ai"])
            st.session_state.chat_title = chat["title"]
            st.rerun()

        new_title = st.text_input("Rename", value=chat["title"], key=f"rename_{chat['timestamp']}")
        if new_title != chat["title"]:
            # Rename the file and update title inside JSON
            old_filename = generate_chat_filename(chat["title"])
            new_filename = generate_chat_filename(new_title)
            user_dir = get_user_log_dir()
            old_path = os.path.join(user_dir, old_filename)
            new_path = os.path.join(user_dir, new_filename)

            if os.path.exists(old_path):
                with open(old_path, "r") as f:
                    data = json.load(f)
                # Security check
                if data.get("user_id") == get_user_id():
                    data["title"] = new_title
                    with open(new_path, "w") as f:
                        json.dump(data, f, indent=2)
                    os.remove(old_path)
                    st.success("Renamed successfully!")
                    st.rerun()

st.sidebar.markdown("""<hr style="border: 1px solid #444;">""", unsafe_allow_html=True)


def start_new_chat():
    st.session_state["chat_history"] = []
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state["chat_title"] = None
    st.rerun()

styled_sidebar_button("New Chat", start_new_chat)


def delete_all_chats_callback():
    delete_all_chats()
    st.session_state["chat_history"] = []
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.success("All chats deleted.")
    st.rerun()

styled_sidebar_button("Delete All Chats", delete_all_chats_callback)


def delete_today_chats_callback():
    delete_today_chats()
    st.success("Deleted today's chats.")
    st.rerun()

styled_sidebar_button("Delete Today's Chats", delete_today_chats_callback)


if "chat_history" in st.session_state and st.session_state.chat_history:
    current_chat_title = st.session_state.get("chat_title", "chat")
    filename = generate_chat_filename(current_chat_title)

    chat_data = {
        "title": current_chat_title,
        "date": get_today_date(),
        "timestamp": get_timestamp(),
        "messages": st.session_state.chat_history,
        "user_id": get_user_id(),
        "uploaded_filename": st.session_state.get("uploaded_filename", None)
    }

    st.sidebar.download_button(
        label="📥 Download Current Chat",
        data=json.dumps(chat_data, indent=2),
        file_name=filename,
        mime="application/json"
    )


# -------------------- Initialize Chat State --------------------
# Initialize user-specific session
initialize_user_session()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------- Main Chat Interface --------------------

# Check if PDF is uploaded and processed
if st.session_state.get("vectordb") is None:
    st.warning("⚠️ Please upload a PDF file to start chatting!")
    st.info("📋 Instructions:")
    st.markdown("""
    1. **Upload a PDF**: Use the file uploader in the sidebar to select your PDF
    2. **Wait for processing**: The system will extract and process the text
    3. **Start chatting**: Ask questions about the content of your PDF
    """)
    st.stop()

# -------------------- Display Chat History --------------------
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("ai"):
        st.markdown(msg["ai"])

# -------------------- Chat Input --------------------
prompt = st.chat_input("Ask a question about your PDF...")

if prompt:
    if st.session_state.get("qa_chain") is None:
        st.error("Please upload a PDF first!")
        st.stop()
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = st.session_state.qa_chain.run(prompt)
        except Exception as e:
            response = f"I apologize, but I encountered an error while processing your question: {str(e)}"

    with st.chat_message("ai"):
        st.markdown(response)

    st.session_state.chat_history.append({"user": prompt, "ai": response})

    # Set title on first message
    if "chat_title" not in st.session_state or not st.session_state.chat_title:
        # Try summarizing by slicing intelligently
        clean_prompt = re.sub(r'\s+', ' ', prompt.strip())
        st.session_state.chat_title = clean_prompt[:40] + ("..." if len(clean_prompt) > 40 else "")

    # Save chat persistently
    save_full_chat(st.session_state.chat_title, st.session_state.chat_history)