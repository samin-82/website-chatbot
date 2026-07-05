# list_models.py
from google import genai
from config import GEMINI_API_KEY

# ایجاد کلاینت
client = genai.Client(api_key=GEMINI_API_KEY)

# دریافت لیست مدل‌ها
print("📋 لیست مدل‌های در دسترس:\n")
print("=" * 60)

try:
    # دریافت لیست مدل‌ها
    models = client.models.list()
    
    # نمایش مدل‌های قابل استفاده برای generateContent
    for model in models:
        # فقط مدل‌هایی که برای تولید محتوا مناسب هستند را نمایش بده
        if "generateContent" in str(model.supported_actions):
            print(f"✅ {model.name}")
            print(f"   {model.display_name if hasattr(model, 'display_name') else ''}")
            print("-" * 40)
            
except Exception as e:
    print(f"❌ خطا در دریافت لیست مدل‌ها: {e}")
    print("\nراهنمایی: مطمئن شو کلید API درست است و به اینترنت دسترسی داری.")