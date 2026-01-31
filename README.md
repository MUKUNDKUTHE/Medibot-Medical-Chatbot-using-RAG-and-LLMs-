MediBot – Medical Chatbot using RAG and LLM

A document-based medical chatbot built using Streamlit and LangChain, powered by Groq's LLaMA-3.3-70B model. MediBot answers medical questions strictly based on a user-uploaded PDF document using Retrieval-Augmented Generation (RAG), ensuring accurate, context-aware, and traceable responses.

Features

Upload medical PDF documents
Ask medical questions based on document content
RAG-based contextual answering
Source document preview with page numbers
Conversational memory
Save and revisit previous chats
Simple Streamlit web interface

Tech Stack
Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, PyPDF2, Groq LLM API, LLaMA 3.3 (70B), Sentence-Transformers

How It Works

User uploads a medical PDF document
Text is extracted from PDF
Text is split into small chunks
Embeddings are generated using HuggingFace Sentence Transformers
FAISS vector store is created for semantic search
User enters a medical question
Relevant chunks are retrieved from FAISS
Retrieved context + question is sent to Groq-hosted LLaMA 3.3
Model generates answer grounded in document context
Answer and source references are displayed
Chat history is stored in session
