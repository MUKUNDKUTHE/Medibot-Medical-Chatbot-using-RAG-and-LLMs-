import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq  

import streamlit as st
os.environ['GROQ_API_KEY'] = st.secrets["groq_api_key"]

DB_FAISS_PATH = "vectorstore/db_faiss"
PDF_PATH = "Medical Book.pdf"

@st.cache_resource
def vector():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = os.path.join(DB_FAISS_PATH, "index.faiss")
    pkl_file = os.path.join(DB_FAISS_PATH, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        pdf_reader = PdfReader(PDF_PATH)
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks = splitter.split_text(page_text)
                for chunk in chunks:
                    docs.append({
                        "page_content": chunk,
                        "metadata": {"page_number": i + 1}
                    })
        texts = [doc["page_content"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vectorstore.save_local(DB_FAISS_PATH)
        return vectorstore 

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  
        temperature=0.5,
    )

def context_history(chat):
    history = chat[-3:]
    context = ""
    for i in history:
        context += f"User: {i['question']}\nAssistant: {i['answer']}\n"
    return context

def main():
    st.title("MediBot - Medical Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "saved_chats" not in st.session_state:
        st.session_state.saved_chats = []

    if "selected_chat_index" not in st.session_state:
        st.session_state.selected_chat_index = None

    with st.sidebar:
        st.markdown("## Chat Options")
        if st.button("Start New Chat"):
            if st.session_state.chat_history:
                st.session_state.saved_chats.append(st.session_state.chat_history.copy())
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None

        if st.session_state.saved_chats:
            chat_options = [f"Chat {i+1} ({len(chat)} messages)" for i, chat in enumerate(st.session_state.saved_chats)]
            selected_index = st.selectbox("Select a chat to view", options=list(range(len(chat_options))), format_func=lambda x: chat_options[x])
            if st.button("View Selected Chat"):
                st.session_state.selected_chat_index = selected_index

    if st.session_state.selected_chat_index is not None:
        active_chat = st.session_state.saved_chats[st.session_state.selected_chat_index]
    else:
        active_chat = st.session_state.chat_history

    for message in active_chat:
        with st.chat_message("user"):
            st.markdown(message["question"])
        with st.chat_message("assistant"):
            st.markdown(message["answer"])

    vector_store = vector()
    llm = load_llm()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a medical assistant. Answer the question based on the context provided. If the question 
is not from the context reply as \"Out of domain question.\".

Context:
{context}

Question:
{question}

Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    question = st.chat_input("Enter your medical related question:")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Getting answer..."):
            ques = f"{context_history(active_chat)} User: {question}"
            result = qa_chain.invoke({"query": ques})
            answer = result["result"]
            sources = result["source_documents"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        active_chat.append({
            "question": question,
            "answer": answer,
        })

        if answer.strip() != "Out of domain question.":
            with st.sidebar:
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page_number", "Unknown")
                    content_words = doc.page_content.split()
                    preview = " ".join(content_words[:40]) + ("..." if len(content_words) > 40 else "")
                    st.markdown(f"**Sources {i+1} (Page {page})**")
                    st.write(preview)
        else:
            with st.sidebar:
                st.markdown("**No sources found.**")

if __name__ == "__main__":
    main()
