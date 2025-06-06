# 🤖 Chat with Gemini

A Streamlit application that combines **Retrieval-Augmented Generation (RAG)** with **Google Gemini AI** to create an intelligent document chat system with multi-modal capabilities.

## ✨ Features

- **📚 RAG Chat**: Upload PDFs and chat with your documents using intelligent retrieval
- **💬 Direct Chat**: Have conversations directly with Google Gemini
- **🖼️ Image Analysis**: Upload and analyze images with Gemini Vision
- **📄 PDF Processing**: Automatic text extraction and intelligent chunking
- **🔍 Source Attribution**: See which documents were used for each response
- **💾 Conversation Memory**: Maintains context across multi-turn conversations

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash
- **Framework**: LangChain
- **Vector Store**: DocArrayInMemorySearch
- **Embeddings**: Google Generative AI Embeddings
- **PDF Processing**: PyPDF2
- **Image Processing**: Pillow + Gemini Vision


## ⚠️ Privacy Notice

- API calls are made to Google's servers
- No data is permanently stored by the application
