# chatbot_app_final.py (نسخه REST API)
import streamlit as st
import pickle
import faiss
import numpy as np
import os
import requests
import json

try:
    from config import GEMINI_API_KEY
except ImportError:
    st.error("⚠️ فایل config.py پیدا نشد!")
    st.stop()

def call_gemini_api(prompt, api_key):
    """
    فراخوانی مستقیم REST API گوگل با مدل‌های موجود
    """
    # استفاده از مدل‌های موجود در لیست
    models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-pro-latest"
    ]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            # اگر خطا بود، مدل بعدی را امتحان کن
        except:
            continue
    
    return None

def load_existing_vector_store():
    index = faiss.read_index("vector_index.faiss")
    with open("chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embedding_model.pkl", "rb") as f:
        model = pickle.load(f)
    return index, chunks, model

def get_answer_with_api(question, index, chunks, model, k=4):
    # تبدیل سوال به بردار
    question_embedding = model.encode([question]).astype('float32')
    distances, indices = index.search(question_embedding, k)
    
    # پیدا کردن تکه‌های مرتبط
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n---\n\n".join([f"منبع: {c['source']}\nمتن: {c['text']}" for c in retrieved_chunks])
    
    # ساخت پرامپت
    prompt = f"""بر اساس متن‌های زیر که از وبسایت استخراج شده‌اند، به سوال کاربر پاسخ بده.
اگر جواب در متن‌ها نبود، بگو "اطلاعاتی در این مورد در وبسایت پیدا نشد".
همیشه در پایان پاسخ، منابع (آدرس صفحات) را ذکر کن.

زمینه (Context):
{context}

سوال: {question}
پاسخ:"""
    
    # دریافت پاسخ از Gemini
    answer = call_gemini_api(prompt, GEMINI_API_KEY)
    
    if answer is None:
        answer = "متاسفانه ارتباط با Gemini برقرار نشد. لطفاً از نسخه ساده استفاده کن."
    
    sources = list(set([c['source'] for c in retrieved_chunks]))
    answer += f"\n\n📚 **منابع**: " + ", ".join(sources)
    return answer

# ---------- رابط کاربری ----------
st.set_page_config(page_title="چت‌بات وبسایت", page_icon="🤖")
st.title("🤖 چت‌بات اختصاصی وبسایت مخابرات ایران")
st.caption("پرسش‌های خود را درباره خدمات و محصولات شرکت مخابرات ایران بپرسید.")

if not (os.path.exists("vector_index.faiss") and os.path.exists("chunks_metadata.pkl")):
    st.error("⚠️ فایل‌های بانک اطلاعاتی پیدا نشد! لطفاً ابتدا فایل `build_index.py` را اجرا کن.")
    st.stop()

@st.cache_resource
def load_data():
    return load_existing_vector_store()

try:
    index, chunks, model = load_data()
except Exception as e:
    st.error(f"خطا در بارگذاری دیتا: {e}")
    st.stop()

question = st.text_input("💬 سوال خود را در مورد وبسایت بپرس:")

if question:
    with st.spinner("در حال جستجو و پردازش..."):
        try:
            answer = get_answer_with_api(question, index, chunks, model)
            st.markdown("### 📝 پاسخ:")
            st.write(answer)
        except Exception as e:
            st.error(f"خطا رخ داد: {e}")


# # chatbot_simple.py
# import streamlit as st
# import pickle
# import faiss
# import numpy as np
# import os

# def load_data():
#     index = faiss.read_index("vector_index.faiss")
#     with open("chunks_metadata.pkl", "rb") as f:
#         chunks = pickle.load(f)
#     with open("embedding_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     return index, chunks, model

# def find_similar(question, index, chunks, model, k=5):
#     question_embedding = model.encode([question]).astype('float32')
#     distances, indices = index.search(question_embedding, k)
#     retrieved_chunks = [chunks[i] for i in indices[0]]
#     return retrieved_chunks

# st.set_page_config(page_title="چت‌بات وبسایت", page_icon="🤖")
# st.title("🤖 چت‌بات وبسایت مخابرات ایران")
# st.caption("مطالب مرتبط با سوال شما از وبسایت پیدا می‌شود.")

# if not (os.path.exists("vector_index.faiss") and os.path.exists("chunks_metadata.pkl")):
#     st.error("⚠️ فایل‌های بانک اطلاعاتی پیدا نشد!")
#     st.stop()

# @st.cache_resource
# def load():
#     return load_data()

# index, chunks, model = load()

# question = st.text_input("💬 سوال خود را بپرس:")

# if question:
#     with st.spinner("در حال جستجو..."):
#         results = find_similar(question, index, chunks, model)
        
#         st.markdown("### 📝 بخش‌های مرتبط پیدا شده:")
#         for i, chunk in enumerate(results, 1):
#             with st.expander(f"📄 منبع {i}: {chunk['source']}"):
#                 st.write(chunk['text'])
#                 st.caption(f"🔗 {chunk['source']}")


