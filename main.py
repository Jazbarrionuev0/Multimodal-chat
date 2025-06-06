import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import io
from PIL import Image
import base64
import time


st.set_page_config(
    page_title="RAG Chat with Gemini: Files & Image",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def initialize_components():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        st.stop()
    
    genai.configure(api_key=google_api_key)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=google_api_key
    )
    
    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=google_api_key
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    return llm, vision_llm, embeddings

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def process_image_with_gemini(image_file, prompt, vision_llm):
    try:
        image = Image.open(image_file)
        
        # base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        )
        
        response = vision_llm.invoke([message])
        return response.content
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}"

def create_vector_store(embeddings, documents):
    if not documents:
        return None
    
    if isinstance(documents[0], str):
        #  strings to objects
        docs = [Document(page_content=doc) for doc in documents if doc.strip()]
    else:
        docs = documents
    
    if not docs:
        return None
    
    vectorstore = DocArrayInMemorySearch.from_documents(
        docs,
        embedding=embeddings
    )
    
    return vectorstore

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def initialize_memory():
    """Initialize conversation memory"""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Header
st.title("🤖 Chat with Gemini")
st.markdown("Chat with AI, upload PDFs, and analyze images")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not os.getenv("GOOGLE_API_KEY"):
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key"
        )
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Chat mode
    st.header("💬 Chat Mode")
    chat_mode = st.selectbox(
        "Select chat mode:",
        ["RAG Chat", "Direct Chat", "Image Analysis"]
    )
    
    st.header("📚 Knowledge Base")
    
    # Pdf upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files to add to the knowledge base"
    )
    
    # Image upload
    st.subheader("🖼️ Upload Images")
    uploaded_images = st.file_uploader(
        "Upload images for analysis",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images for analysis with Gemini Vision"
    )
    

# components
try:
    llm, vision_llm, embeddings = initialize_components()
    
    # uploaded PDFs
    pdf_texts = []
    if uploaded_files:
        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)
                if text:
                    # Chunk the text
                    chunks = chunk_text(text)
                    pdf_texts.extend(chunks)
                    st.sidebar.success(f"✅ Processed {pdf_file.name}")
 
    
    # vector store for RAG mode
    vectorstore = None
    qa_chain = None
    
    if chat_mode == "RAG Chat" and pdf_texts:
        vectorstore = create_vector_store(embeddings, pdf_texts)
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # memory
            if 'memory' not in st.session_state:
                st.session_state.memory = initialize_memory()
            
            # RAG chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
                verbose=True
            )
    
    st.sidebar.info(f"📄 Total documents in knowledge base: {len(pdf_texts)}")
    
    if chat_mode == "RAG Chat":
        st.info("🔍 RAG Mode: AI will search the knowledge base to answer your questions")
    elif chat_mode == "Direct Chat":
        st.info("💬 Direct Mode: Chat directly with Gemini (no document retrieval)")
    else:
        st.info("🖼️ Image Analysis Mode: Upload images and ask questions about them")
    
except Exception as e:
    st.error(f"Error initializing components: {str(e)}")
    st.stop()

# chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display images
if uploaded_images and chat_mode == "Image Analysis":
    st.text(" Uploaded Images:")
    cols = st.columns(min(len(uploaded_images), 3))
    for i, img_file in enumerate(uploaded_images):
        with cols[i % 3]:
            image = Image.open(img_file)
            st.image(image, width=200)
    st.markdown("---")

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {source}")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if chat_mode == "RAG Chat" and qa_chain:
                    # RAG mode: use kb
                    response = qa_chain({"question": prompt})
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])
                    sources = [doc.page_content for doc in source_docs]
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                st.write(f"**Source {i+1}:** {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                
                elif chat_mode == "Direct Chat":
                    # Direct chat mode
                    response = llm.invoke(prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                
                elif chat_mode == "Image Analysis" and uploaded_images:
                    # Image analysis mode
                    answers = []
                    for img_file in uploaded_images:
                        img_response = process_image_with_gemini(img_file, prompt, vision_llm)
                        answers.append(f"**Analysis of {img_file.name}:**\n{img_response}")
                    
                    combined_answer = "\n\n---\n\n".join(answers)
                    st.markdown(combined_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": combined_answer
                    })
                
                else:
                    response = llm.invoke(prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Clear chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    if 'memory' in st.session_state:
        st.session_state.memory = initialize_memory()
    st.rerun()

# Instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    **RAG Chat Mode:**
    - Upload PDFs or add custom documents
    - AI retrieves relevant info from knowledge base
    
    **Direct Chat Mode:**
    - Chat directly with Gemini
    - No document retrieval
    
    **Image Analysis Mode:**
    - Upload images (PNG, JPG, JPEG)
    - Ask questions about the images
    - Gemini Vision will analyze them
    """)
    