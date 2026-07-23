import time
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_URL = "https://irvex.ir/"
MAX_PAGES = 30
DELAY = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fa,en;q=0.5",
}


def crawl(start_url, max_pages=30, delay=1):
    visited = set()
    to_visit = [start_url]
    all_texts = []

    print(f"Starting crawl from {start_url}...")

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            print(f"Fetching: {current_url}")
            resp = requests.get(current_url, headers=HEADERS, timeout=15, allow_redirects=True)
            if resp.status_code != 200:
                print(f"  Skipped (status {resp.status_code})")
                visited.add(current_url)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())

            if len(text) > 200:
                all_texts.append({"url": current_url, "content": text})
                print(f"  OK ({len(text)} chars)")
            else:
                print(f"  Skipped (too short: {len(text)} chars)")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("/"):
                    href = urljoin(current_url, href)
                if href.startswith("http") and urlparse(href).netloc == urlparse(start_url).netloc:
                    if "#" in href:
                        href = href.split("#")[0]
                    if href not in visited and href not in to_visit:
                        to_visit.append(href)

        except Exception as e:
            print(f"  Error: {e}")

        visited.add(current_url)
        time.sleep(delay)

    print(f"Crawl complete. {len(all_texts)} pages with content.")
    return all_texts


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in docs:
        for chunk_text in splitter.split_text(doc["content"]):
            if len(chunk_text) > 50:
                chunks.append({"text": chunk_text, "source": doc["url"]})
    print(f"Created {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"Index built. Vectors: {index.ntotal}")

    faiss.write_index(index, "vector_index.faiss")
    with open("chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("embedding_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("All files saved.")
    return index, chunks, model


if __name__ == "__main__":
    print("Step 1: Crawling website...")
    docs = crawl(TARGET_URL, max_pages=MAX_PAGES, delay=DELAY)

    if docs:
        print("Step 2: Splitting into chunks...")
        chunks = chunk_documents(docs)
        if chunks:
            print("Step 3: Building vector index...")
            build_vector_store(chunks)
            print("\nDone! Now run the chatbot.")
        else:
            print("No chunks created. Pages may have empty content.")
    else:
        print("No content found. Check the URL.")
