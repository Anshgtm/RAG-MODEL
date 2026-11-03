import os
import hashlib
import streamlit as st

from dotenv import load_dotenv
from io import BytesIO

from pypdf import PdfReader
from docx import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

st.set_page_config(
    page_title="RAG-MODEL",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind AI")
st.caption("Upload your document and ask anything about it.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please configure your .env file.")
    st.stop()


# ---------------- MODELS ----------------

@st.cache_resource
def load_models():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    return embedding_model, llm


embedding_model, llm = load_models()


# ---------------- DOCUMENT EXTRACTION ----------------

def extract_document(file):

    file_bytes = file.getvalue()
    extension = file.name.split(".")[-1].lower()

    documents = []

    if extension == "pdf":

        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "page": page_number + 1
                })

    elif extension == "docx":

        doc = Document(BytesIO(file_bytes))

        text = "\n".join(
            paragraph.text for paragraph in doc.paragraphs
        )

        if text.strip():
            documents.append({
                "text": text,
                "page": 1
            })

    elif extension in ["txt", "md"]:

        text = file_bytes.decode("utf-8-sig")

        if text.strip():
            documents.append({
                "text": text,
                "page": 1
            })

    return documents


# ---------------- VECTOR DATABASE ----------------

def create_vectorstore(file):

    extracted_data = extract_document(file)

    if not extracted_data:
        raise ValueError(
            "No readable text found in this document."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for item in extracted_data:

        split_docs = splitter.create_documents(
            texts=[item["text"]],
            metadatas=[{
                "source": file.name,
                "page": item["page"]
            }]
        )

        chunks.extend(split_docs)

    # Unique collection for each uploaded document
    file_hash = hashlib.sha256(
        file.getvalue()
    ).hexdigest()

    collection_name = "doc_" + file_hash[:24]

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory="chroma_db"
    )

    return vectorstore, file_hash


# ---------------- RAG PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are DocuMind AI, a helpful document assistant.

        Answer the user's question using ONLY the
        information available in the provided context.

        Rules:
        1. Give clear and detailed answers.
        2. Explain concepts in simple language.
        3. Use bullet points when appropriate.
        4. Do not make up information.
        5. If the answer is not available in the
           document, say:

           "I could not find the answer in the document."

        6. Conversation history can help understand
           follow-up questions, but factual answers
           must come only from the document context.
        """
    ),

    (
        "human",
        """
        Conversation History:
        {chat_history}

        Document Context:
        {context}

        User Question:
        {question}

        Answer:
        """
    )

])


# ---------------- QUESTION ANSWERING ----------------

def get_answer(vectorstore, question, chat_history):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"Page {doc.metadata.get('page', 'Unknown')}:\n"
        f"{doc.page_content}"
        for doc in docs
    ])

    final_prompt = prompt.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    response = llm.invoke(final_prompt)

    return response.content, docs


# ---------------- SESSION STATE ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose your document",
        type=["pdf", "docx", "txt", "md"]
    )

    if uploaded_file:

        current_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        if current_hash != st.session_state.file_hash:

            if st.button("Process Document", type="primary"):

                try:

                    with st.spinner(
                        "Reading document and creating embeddings..."
                    ):

                        vectorstore, file_hash = create_vectorstore(
                            uploaded_file
                        )

                        st.session_state.vectorstore = vectorstore
                        st.session_state.file_hash = file_hash
                        st.session_state.messages = []

                    st.success("Document processed successfully!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        else:
            st.success("Document already processed!")

    if st.session_state.vectorstore:

        st.divider()

        st.success("Document is ready for questions.")

        if st.button("Clear Chat"):

            st.session_state.messages = []
            st.rerun()


# ---------------- MAIN CHAT UI ----------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a document from the sidebar "
        "and click Process Document to get started."
    )

    st.markdown("### What can you do?")
    st.write("- Summarize your document")
    st.write("- Ask questions about any topic")
    st.write("- Explain difficult concepts")
    st.write("- Find important information")
    st.write("- Ask follow-up questions")

else:

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("View Sources"):

                    for source in message["sources"]:
                        st.write(source)

    # Chat input

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Build history from previous messages
        history = "\n".join([
            f'{msg["role"]}: {msg["content"]}'
            for msg in st.session_state.messages[-7:-1]
        ])

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer, docs = get_answer(
                        st.session_state.vectorstore,
                        question,
                        history
                    )

                    st.markdown(answer)

                    sources = list(dict.fromkeys([
                        f'{doc.metadata.get("source", "Document")} '
                        f'— Page {doc.metadata.get("page", "Unknown")}'
                        for doc in docs
                    ]))

                    with st.expander("View Sources"):
                        for source in sources:
                            st.write("📄", source)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    error_message = f"Something went wrong: {str(e)}"

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
