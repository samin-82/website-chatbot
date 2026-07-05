import time
import pickle
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_URL = "https://irvex.ir/"
MAX_PAGES = 30  
DELAY = 2

def setup_driver():
    """تنظیم مرورگر Chrome با WebDriver Manager"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def crawl_dynamic(start_url, max_pages=20, delay=2):
    """خزش وبسایت‌های داینامیک با Selenium"""
    driver = setup_driver()
    visited = set()
    to_visit = [start_url]
    all_texts = []
    
    print(f"شروع خزش از {start_url} با Selenium و WebDriver Manager...")
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        try:
            print(f"در حال باز کردن: {current_url}")
            driver.get(current_url)
            
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(delay)
            except:
                print(f"⏰ تایم‌اوت در بارگذاری {current_url}")
            
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            
            if len(text) > 200:
                all_texts.append({"url": current_url, "content": text})
                print(f"✅ خزیده شد: {current_url} (طول: {len(text)})")
            else:
                print(f"⚠️ متن کوتاه در: {current_url} (طول: {len(text)})")
            
            links = driver.find_elements(By.TAG_NAME, "a")
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and href.startswith("http"):
                        if urlparse(href).netloc == urlparse(start_url).netloc:
                            if href not in visited and href not in to_visit:
                                to_visit.append(href)
                except:
                    continue
                            
        except Exception as e:
            print(f"❌ خطا در {current_url}: {e}")
            
        visited.add(current_url)
        time.sleep(1)
        
    driver.quit()
    print(f"خزش تمام شد. {len(all_texts)} صفحه با محتوا پیدا شد.")
    return all_texts

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in docs:
        chunks_text = splitter.split_text(doc["content"])
        for chunk in chunks_text:
            if len(chunk) > 50:
                chunks.append({"text": chunk, "source": doc["url"]})
    print(f"تعداد تکه‌های ایجاد شده: {len(chunks)}")
    return chunks

def build_vector_store(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    print("مدل بردارساز بارگذاری شد.")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"ایندکس ساخته شد. تعداد بردارها: {index.ntotal}")
    
    faiss.write_index(index, "vector_index.faiss")
    with open("chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("embedding_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ تمام فایل‌های بانک اطلاعاتی ذخیره شدند.")
    return index, chunks, model

if __name__ == "__main__":
    print("🚀 مرحله 1: در حال خزش وبسایت با Selenium...")
    docs = crawl_dynamic(TARGET_URL, max_pages=MAX_PAGES, delay=DELAY)
    
    if docs:
        print("🚀 مرحله 2: در حال خرد کردن متن‌ها...")
        chunks = chunk_documents(docs)
        if chunks:
            print("🚀 مرحله 3: در حال ساخت بردارها...")
            build_vector_store(chunks)
            print("\n🎉 تمام شد! حالا چت‌بات را اجرا کن.")
        else:
            print("❌ هیچ تکه‌ای ساخته نشد. محتوای صفحات خالی است.")
    else:
        print("❌ هیچ محتوایی پیدا نشد. آدرس را بررسی کن.")