import streamlit as st
import pickle
import faiss
import numpy as np
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

try:
    from config import NARA_API_KEY, NARA_BASE_URL, NARA_MODEL
except ImportError:
    st.error("config.py not found!")
    st.stop()


@st.cache_resource
def load_vector_store():
    index = faiss.read_index("vector_index.faiss")
    with open("chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embedding_model.pkl", "rb") as f:
        model = pickle.load(f)
    return index, chunks, model


def retrieve(query: str, index, chunks, model, k: int = 5) -> list[dict]:
    embedding = model.encode([query]).astype("float32")
    _, indices = index.search(embedding, k)
    return [chunks[i] for i in indices[0]]


def format_docs(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(f"Source: {d['source']}\nText: {d['text']}" for d in docs)


def get_sources(docs: list[dict]) -> list[str]:
    return list({d["source"] for d in docs})


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on website content.
Answer in the SAME LANGUAGE as the question (Persian if the question is in Persian).
If the answer is not in the provided context, say "I could not find this information on the website."
Always list the source URLs at the end.

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(
    model=NARA_MODEL,
    api_key=NARA_API_KEY,
    base_url=NARA_BASE_URL,
    temperature=0,
)


def build_rag_chain(index, chunks, model):
    def retrieve_and_format(query):
        docs = retrieve(query, index, chunks, model)
        return format_docs(docs)

    chain = (
        {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


st.set_page_config(page_title="Website Chatbot", page_icon="🤖")
st.title("🤖 Website Content Chatbot")
st.caption("Ask questions about the website content")

if not os.path.exists("vector_index.faiss") or not os.path.exists("chunks_metadata.pkl"):
    st.error("Vector index not found! Run `build_index.py` first.")
    if st.button("Run Crawler"):
        with st.spinner("Crawling website..."):
            os.system("python build_index.py")
        st.success("Done! Refresh the page.")
    st.stop()

try:
    index, chunks, model = load_vector_store()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.info(f"Loaded {len(chunks)} text chunks from the website.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = build_rag_chain(index, chunks, model)

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask a question about the website:"):
    st.chat_message("user").write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Searching and generating answer..."):
        try:
            chain = st.session_state.rag_chain
            docs = retrieve(question, index, chunks, model)
            answer = chain.invoke(question)
            sources = get_sources(docs)
            if sources:
                answer += f"\n\n**Sources:** {', '.join(sources)}"
        except Exception as e:
            answer = f"Error: {e}"

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
