# smart_chatbot.py
import streamlit as st
import pickle
import faiss
import numpy as np
import os
import requests
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

try:
    from config import GEMINI_API_KEY
except ImportError:
    st.error("⚠️ فایل config.py پیدا نشد! لطفاً کلید API را در آن قرار دهید.")
    st.stop()

# ============================================
# بخش 1: تنظیمات پایه
# ============================================
BASE_URL = "https://irvex.ir/api/v1"  # آدرس API - در صورت نیاز تغییر بده
HEADERS = {"Accept-Language": "fa"}

# ============================================
# بخش 2: توابع فراخوانی API (بر اساس اندپوینت‌های وبسایت)
# ============================================

def fetch_landing_page_data():
    """دریافت اطلاعات صفحه اصلی"""
    try:
        response = requests.get(f"{BASE_URL}/tournaments/landing-page-report", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_tournaments_statistics():
    """دریافت آمار مسابقات"""
    try:
        response = requests.get(f"{BASE_URL}/tournaments/reports/statistics", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_tournaments(time_status="ALL", page=1, page_size=10):
    """دریافت لیست مسابقات با فیلتر"""
    try:
        params = {"page": page, "pageSize": page_size}
        if time_status != "ALL":
            params["timeStatus"] = time_status
        response = requests.get(f"{BASE_URL}/tournaments", headers=HEADERS, params=params, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_tournament_detail(tournament_id):
    """دریافت اطلاعات یک مسابقه خاص"""
    try:
        response = requests.get(f"{BASE_URL}/tournaments/{tournament_id}", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_tournament_leaderboard(tournament_id):
    """دریافت جدول رتبه‌بندی یک مسابقه"""
    try:
        response = requests.get(f"{BASE_URL}/tournaments/{tournament_id}/leaderboard", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_honor_leaderboard():
    """دریافت جدول افتخارات"""
    try:
        response = requests.get(f"{BASE_URL}/achievements/leaderboard", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

def fetch_capital_chart(tournament_id):
    """دریافت نمودار عملکرد یک مسابقه"""
    try:
        response = requests.get(f"{BASE_URL}/analytics/tournaments/{tournament_id}/capital-percentage-chart", headers=HEADERS, timeout=10)
        if response.ok:
            return response.json()
        return None
    except:
        return None

# ============================================
# بخش 3: توابع کمکی برای جستجو در دیتابیس محلی
# ============================================

def load_vector_store():
    """بارگذاری دیتابیس محلی"""
    index = faiss.read_index("vector_index.faiss")
    with open("chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embedding_model.pkl", "rb") as f:
        model = pickle.load(f)
    return index, chunks, model

def search_local_db(question, index, chunks, model, k=5):
    """جستجو در دیتابیس محلی"""
    question_embedding = model.encode([question]).astype('float32')
    distances, indices = index.search(question_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# ============================================
# بخش 4: تابع تشخیص نیت و پردازش درخواست
# ============================================

def call_gemini_api(prompt):
    """فراخوانی مستقیم Gemini API"""
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro-latest"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            continue
    return None

def detect_intent(user_input):
    """تشخیص نیت کاربر با استفاده از Gemini"""
    prompt = f"""
    متن کاربر: "{user_input}"
    
    قابلیت‌های موجود:
    1. landing_data - مشاهده اطلاعات صفحه اصلی و آمار کلی
    2. tournaments_stats - مشاهده آمار مسابقات (تعداد جاری، آینده، تمام شده)
    3. tournaments_list - مشاهده لیست مسابقات (با فیلتر UPCOMING, ONGOING, ENDED)
    4. tournament_detail - مشاهده جزئیات یک مسابقه خاص (نیاز به ID)
    5. tournament_leaderboard - مشاهده جدول رتبه‌بندی یک مسابقه (نیاز به ID)
    6. honor_leaderboard - مشاهده جدول افتخارات کلی
    7. general_question - سوال عمومی که نیاز به جستجو در محتوای وبسایت دارد
    
    فقط نام قابلیت را برگردان. اگر هیچکدام مطابقت نداشت، "general_question" را برگردان.
    """
    
    response = call_gemini_api(prompt)
    if response:
        return response.strip().lower()
    return "general_question"

def extract_tournament_id(user_input):
    """استخراج ID مسابقه از متن کاربر"""
    prompt = f"""
    متن کاربر: "{user_input}"
    
    اگر کاربر به یک مسابقه خاص اشاره کرده، ID آن مسابقه را استخراج کن.
    فقط ID را برگردان، اگر نبود "none" را برگردان.
    """
    response = call_gemini_api(prompt)
    if response and response.strip().lower() != "none":
        return response.strip()
    return None

def extract_time_filter(user_input):
    """استخراج فیلتر زمان از متن کاربر"""
    user_lower = user_input.lower()
    if "آینده" in user_lower or "upcoming" in user_lower:
        return "UPCOMING"
    elif "جاری" in user_lower or "در حال" in user_lower or "ongoing" in user_lower:
        return "ONGOING"
    elif "تمام" in user_lower or "گذشته" in user_lower or "ended" in user_lower:
        return "ENDED"
    return "ALL"

def execute_api_call(intent, params):
    """اجرای تابع API بر اساس نیت تشخیص داده شده"""
    
    functions = {
        "landing_data": fetch_landing_page_data,
        "tournaments_stats": fetch_tournaments_statistics,
        "honor_leaderboard": fetch_honor_leaderboard,
    }
    
    if intent in functions:
        return functions[intent]()
    
    if intent == "tournaments_list":
        return fetch_tournaments(
            time_status=params.get("time_status", "ALL"),
            page=1,
            page_size=10
        )
    
    if intent == "tournament_detail":
        tournament_id = params.get("tournament_id")
        if tournament_id:
            return fetch_tournament_detail(tournament_id)
        return {"error": "شناسه مسابقه مشخص نشده است"}
    
    if intent == "tournament_leaderboard":
        tournament_id = params.get("tournament_id")
        if tournament_id:
            return fetch_tournament_leaderboard(tournament_id)
        return {"error": "شناسه مسابقه مشخص نشده است"}
    
    return {"error": "قابلیت مورد نظر شناسایی نشد"}

def format_api_response(data):
    """فرمت‌دهی پاسخ API برای نمایش بهتر"""
    if not data:
        return "اطلاعاتی در این مورد در وبسایت پیدا نشد."
    
    if isinstance(data, dict):
        if "error" in data:
            return f"⚠️ {data['error']}"
        
        if "data" in data:
            data = data["data"]
    
    return json.dumps(data, ensure_ascii=False, indent=2)

# ============================================
# بخش 5: تابع اصلی پردازش
# ============================================

def process_user_request(user_input, index, chunks, model):
    """پردازش درخواست کاربر و تولید پاسخ"""
    
    # 1. تشخیص نیت
    intent = detect_intent(user_input)
    
    # 2. استخراج پارامترها
    params = {}
    tournament_id = extract_tournament_id(user_input)
    if tournament_id:
        params["tournament_id"] = tournament_id
    
    if intent == "tournaments_list":
        params["time_status"] = extract_time_filter(user_input)
    
    # 3. اگر نیت مربوط به API است
    if intent != "general_question":
        result = execute_api_call(intent, params)
        
        if result:
            formatted_result = format_api_response(result)
            
            # تولید پاسخ نهایی با Gemini
            response_prompt = f"""
            سوال کاربر: "{user_input}"
            نتیجه دریافتی از وبسایت: {formatted_result}
            
            یک پاسخ مفید و طبیعی به کاربر بده. اگر اطلاعاتی پیدا نشد، بگو "اطلاعاتی در این مورد در وبسایت پیدا نشد".
            """
            final_response = call_gemini_api(response_prompt)
            if final_response:
                return final_response
            return formatted_result
    
    # 4. اگر نیت عمومی است، از دیتابیس محلی استفاده کن
    retrieved_chunks = search_local_db(user_input, index, chunks, model)
    context = "\n\n---\n\n".join([f"منبع: {c['source']}\nمتن: {c['text']}" for c in retrieved_chunks])
    
    prompt = f"""بر اساس متن‌های زیر که از وبسایت استخراج شده‌اند، به سوال کاربر پاسخ بده.
اگر جواب در متن‌ها نبود، بگو "اطلاعاتی در این مورد در وبسایت پیدا نشد".
همیشه در پایان پاسخ، منابع (آدرس صفحات) را ذکر کن.

زمینه (Context):
{context}

سوال: {user_input}
پاسخ:"""
    
    answer = call_gemini_api(prompt)
    if answer is None:
        answer = "متاسفانه ارتباط با Gemini برقرار نشد. لطفاً از نسخه ساده استفاده کن."
    
    sources = list(set([c['source'] for c in retrieved_chunks]))
    answer += f"\n\n📚 **منابع**: " + ", ".join(sources)
    
    return answer

# ============================================
# بخش 6: رابط کاربری Streamlit
# ============================================

st.set_page_config(page_title="چت‌بات هوشمند", page_icon="🏆")
st.title("🏆 چت‌بات هوشمند")
st.caption("سوالات خود را بپرسید - من می‌توانم از وبسایت پاسخ بگیرم!")

# نمایش قابلیت‌ها
with st.expander("📋 من می‌توانم به شما کمک کنم:"):
    st.markdown("""
    - 📊 **آمار مسابقات**: تعداد مسابقات جاری، آینده و تمام شده
    - 📋 **لیست مسابقات**: مشاهده مسابقات با فیلترهای مختلف
    - 🏅 **جدول رتبه‌بندی**: دیدن نتایج و رتبه‌بندی مسابقات
    - 🏆 **جدول افتخارات**: مشاهده برترین‌های کلی
    - ℹ️ **جزئیات مسابقه**: اطلاعات کامل یک مسابقه خاص
    - 🔍 **سوالات عمومی**: جستجو در محتوای وبسایت
    """)

# بررسی وجود فایل‌های دیتابیس
if not (os.path.exists("vector_index.faiss") and os.path.exists("chunks_metadata.pkl")):
    st.warning("⚠️ فایل‌های بانک اطلاعاتی پیدا نشد! لطفاً ابتدا فایل `build_index.py` را اجرا کن.")
    
    if st.button("🔄 اجرای خزش"):
        with st.spinner("در حال خزش وبسایت..."):
            os.system("python build_index.py")
        st.success("خزش کامل شد! صفحه را Refresh کن.")
    st.stop()

# بارگذاری دیتابیس
@st.cache_resource
def load_data():
    return load_vector_store()

try:
    index, chunks, model = load_data()
except Exception as e:
    st.error(f"خطا در بارگذاری دیتا: {e}")
    st.stop()

st.info(f"✅ {len(chunks)} تکه اطلاعات از وبسایت بارگذاری شد.")

# ورودی کاربر
question = st.text_input("💬 سوال خود را بپرسید:")

if question:
    with st.spinner("در حال پردازش..."):
        try:
            answer = process_user_request(question, index, chunks, model)
            st.markdown("### 📝 پاسخ:")
            st.write(answer)
        except Exception as e:
            st.error(f"خطا رخ داد: {e}")
            st.info("لطفاً سوال خود را واضح‌تر بپرسید.")