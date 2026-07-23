# مستندات کامل پروژه‌های هوش مصنوعی کاربردی

---

## فهرست مطالب

1. [مقدمه](#مقدمه)
2. [پروژه ۱: چت‌بات وبسایت](#پروژه-۱-چت‌بات-وبسایت)
3. [پروژه ۲: سیستم ایجنت هوشمند](#پروژه-۲-سیستم-ایجنت-هوشمند)
4. [ساختار فایل‌ها](#ساختار-فایل‌ها)
5. [نحوه اجرا](#نحوه-اجرا)
6. [نتیجه‌گیری](#نتیجه‌گیری)

---

## مقدمه

این مستند شامل توضیحات کامل دو پروژه هوش مصنوعی کاربردی است:

- **پروژه ۱**: چت‌بات وبسایت مبتنی بر RAG (بازیابی تقویت‌شده تولید) که محتوای وبسایت را خزش کرده و به سوالات کاربران پاسخ می‌دهد.
- **پروژه ۲**: سیستم ایجنت هوشمند با استفاده از LangChain که جایگزین اپراتورهای انسانی می‌شود و API‌های مرتبط را پیدا و فراخوانی می‌کند.

### تکنولوژی‌های استفاده شده

| تکنولوژی | نسخه | کاربرد |
|-----------|-------|--------|
| Python | 3.14 | زبان برنامه‌نویسی اصلی |
| LangChain | 1.3.11 | فریم‌ورک ایجنت و chain |
| LangChain-OpenAI | - | اتصال به API سازگار با OpenAI |
| Streamlit | - | رابط کاربری وب |
| FAISS | - | جستجوی برداری |
| Sentence-Transformers | - | تبدیل متن به بردار |
| BeautifulSoup | - | تحلیل HTML |
| Requests | - | درخواست HTTP |

---

## پروژه ۱: چت‌بات وبسایت

### توضیحات کلی

یک سیستم چت‌بات که وبسایت کارفرما را crawl کرده، محتوای آن را ذخیره می‌کند و کاربران از طریق آن سوالات خود را از محتوای وبسایت می‌پرسند.

### معماری سیستم

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  وبسایت     │────▶│  خزنده       │────▶│  بردارساز   │────▶│  چت‌بات      │
│  (irvex.ir) │     │  (Crawler)   │     │  (Embedder) │     │  (RAG Chain) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │  کاربر       │
                                                              │  (Streamlit) │
                                                              └──────────────┘
```

### مراحل کار

#### مرحله ۱: خزش وبسایت (Crawling)

فایل `build_index.py` مسئول خزش وبسایت است:

1. **شروع از صفحه اصلی**: خزنده از `https://irvex.ir/` شروع می‌کند
2. **استخراج لینک‌ها**: تمام لینک‌های داخلی صفحه را پیدا می‌کند
3. **بازدید از صفحات**: هر صفحه را باز کرده و محتوای متنی آن را استخراج می‌کند
4. **حذف تگ‌های اضافی**: تگ‌های `script`، `style` و `noscript` حذف می‌شوند (تگ‌های `footer`، `header` و `nav` نگه داشته می‌شوند)
5. **ذخیره محتوا**: محتوای هر صفحه به همراه URL ذخیره می‌شود

```python
# کد خزش وبسایت
def crawl(start_url, max_pages=30, delay=1):
    visited = set()
    to_visit = [start_url]
    all_texts = []
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        resp = requests.get(current_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # حذف تگ‌های اضافی (نه footer/header)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 200:
            all_texts.append({"url": current_url, "content": text})
        
        # پیدا کردن لینک‌های جدید
        for a_tag in soup.find_all("a", href=True):
            # اضافه کردن لینک‌های داخلی به صف
            ...
```

#### مرحله ۲: تقسیم متن به بخش‌ها (Chunking)

محتوای استخراج شده با استفاده از `RecursiveCharacterTextSplitter` به بخش‌های کوچکتر تقسیم می‌شود:

- **اندازه هر بخش**: 500 کاراکتر
- **همپوشانی**: 50 کاراکتر
- **جداکننده‌ها**: `\n\n`، `\n`، فاصله

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
```

#### مرحله ۳: تبدیل به بردار (Embedding)

هر بخش متنی با استفاده از مدل `all-MiniLM-L6-v2` به بردار تبدیل می‌شود:

- **مدل**: `SentenceTransformer('all-MiniLM-L6-v2')`
- **ابعاد بردار**: 384
- **ذخیره**: در فایل `embedding_model.pkl`

#### مرحله ۴: ساخت ایندکس FAISS

بردارها در یک ایندکس FAISS ذخیره می‌شوند:

```python
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "vector_index.faiss")
```

#### مرحله ۵: زنجیره RAG (Retrieval-Augmented Generation)

وقتی کاربر سوال می‌پرسد:

1. **تبدیل سوال به بردار**: سوال کاربر با همان مدل به بردار تبدیل می‌شود
2. **جستجوی مشابه**: ۵ بخش مشابه از ایندکس FAISS پیدا می‌شود
3. **ساخت پرامپت**: سوال + زمینه (بخش‌های پیدا شده) در یک پرامپت قرار می‌گیرد
4. **تولید پاسخ**: مدل زبانی (DeepSeek) پاسخ را تولید می‌کند

```python
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on website content.
Answer in the SAME LANGUAGE as the question.
If the answer is not in the provided context, say "I could not find this information."

Context:
{context}

Question: {question}

Answer:"""
)

chain = (
    {"context": retrieve_and_format, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

### فایل‌های پروژه ۱

| فایل | توضیح |
|------|-------|
| `build_index.py` | خزش وبسایت و ساخت ایندکس برداری |
| `project1_chatbot.py` | رابط کاربری چت‌بات با Streamlit |
| `vector_index.faiss` | ایندکس برداری FAISS |
| `chunks_metadata.pkl` | اطلاعات بخش‌های متنی |
| `embedding_model.pkl` | مدل تبدیل به بردار |

### فایل‌های خروجی

- `vector_index.faiss`: ایندکس برداری که توسط FAISS ذخیره می‌شود
- `chunks_metadata.pkl`: لیست بخش‌های متنی به همراه URL منبع
- `embedding_model.pkl`: مدل SentenceTransformer که برای تبدیل سوال به بردار استفاده می‌شود

---

## پروژه ۲: سیستم ایجنت هوشمند

### توضیحات کلی

یک سیستم ایجنت هوشمند که جایگزین اپراتورهای انسانی می‌شود. کاربر یک درخواست می‌دهد و ایجنت API مرتبط را پیدا کرده و فراخوانی می‌کند.

### مفهوم ایجنت

ایجنت در واقع یک مدل زبانی است که به ابزارهایی (tools) مجهز شده است. وقتی کاربر سوالی می‌پرسد:

1. ایجنت **نیت کاربر** را تشخیص می‌دهد
2. **ابزار مناسب** را انتخاب می‌کند
3. **API مرتبط** را فراخوانی می‌کند
4. **پاسخ مناسب** را به کاربر برمی‌گرداند

### معماری سیستم

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  کاربر       │────▶│  ایجنت       │────▶│  ابزارها    │────▶│  API وبسایت  │
│  (سوال)     │     │  (LangChain) │     │  (Tools)    │     │  (irvex.ir)  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  مدل زبانی   │
                    │  (DeepSeek)  │
                    └──────────────┘
```

### ابزارهای موجود (Tools)

| ابزار | توضیح | زمان استفاده |
|-------|-------|-------------|
| `get_landing_page_data` | دریافت اطلاعات صفحه اصلی | سوال درباره پلتفرم |
| `get_tournaments_statistics` | دریافت آمار مسابقات | سوال درباره آمار |
| `get_tournaments_list` | دریافت لیست مسابقات | مشاهده مسابقات |
| `get_tournament_detail` | دریافت جزئیات مسابقه | سوال درباره مسابقه خاص |
| `get_tournament_leaderboard` | دریافت جدول رتبه‌بندی | سوال درباره رتبه‌بندی |
| `get_honor_leaderboard` | دریافت جدول افتخارات | سوال درباره برترین‌ها |
| `get_capital_percentage_chart` | دریافت نمودار عملکرد | سوال درباره نمودار |

### نحوه کار ابزارها

هر ابزار یک تابع Python است که:

1. **دکوراتور `@tool`** دارد
2. **docstring** دارد که توضیح می‌دهد کی و چرا استفاده شود
3. **API مرتبط** را فراخوانی می‌کند
4. **نتیجه** را به صورت JSON برمی‌گرداند

```python
@tool
def get_tournaments_statistics() -> str:
    """Get statistics about tournaments: count of ongoing, upcoming, ended tournaments.
    Use this when the user asks about tournament statistics."""
    try:
        resp = requests.get(f"{BASE_URL}/tournaments/reports/statistics", 
                          headers=HEADERS, timeout=10)
        if resp.ok:
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {e}"
    return "Could not fetch tournament statistics."
```

### نحوه کار ایجنت

```python
# ایجاد ایجنت با LangChain
agent = create_agent(
    model=llm,          # مدل زبانی (DeepSeek)
    tools=TOOLS,        # لیست ابزارها
    system_prompt=SYSTEM_PROMPT,  # دستورالعمل سیستم
)

# فراخوانی ایجنت
response = agent.invoke({
    "messages": [HumanMessage(content="آمار مسابقات چقدر است؟")]
})
```

### پرامپت سیستم

ایجنت با یک پرامپت سیستم راه‌اندازی می‌شود که رفتار آن را مشخص می‌کند:

```
You are a helpful AI assistant for the irvex.ir tournament platform.
You can answer questions about tournaments, leaderboards, platform statistics, and more.

When the user asks a question:
1. Determine which tool(s) to use based on the question.
2. Call the appropriate tool(s) to get the data.
3. Format the response clearly and in the SAME LANGUAGE as the question.
4. If the tool returns JSON data, present it in a readable, user-friendly format.
```

### فایل‌های پروژه ۲

| فایل | توضیح |
|------|-------|
| `project2_agent.py` | ایجنت هوشمند با ابزارها |
| `config.py` | کلید API و تنظیمات |

---

## ساختار فایل‌ها

```
project1/
├── config.py              # تنظیمات (کلید API)
├── requirements.txt       # پکیج‌های مورد نیاز
├── build_index.py         # خزنده وبسایت + ساخت ایندکس
├── project1_chatbot.py    # چت‌بات وبسایت (پروژه ۱)
├── project2_agent.py      # ایجنت هوشمند (پروژه ۲)
├── DOCUMENTATION.md       # این فایل مستندات
├── vector_index.faiss     # ایندکس برداری
├── chunks_metadata.pkl    # اطلاعات بخش‌های متنی
└── embedding_model.pkl    # مدل تبدیل به بردار
```

---

## نحوه اجرا

### پیش‌نیازها

```bash
# نصب پکیج‌ها
pip install -r requirements.txt
```

### پروژه ۱: چت‌بات وبسایت

```bash
# مرحله ۱: خزش وبسایت و ساخت ایندکس
python build_index.py

# مرحله ۲: اجرای چت‌بات
streamlit run project1_chatbot.py
```

### پروژه ۲: ایجنت هوشمند

```bash
# اجرای ایجنت
streamlit run project2_agent.py
```

---

## نتیجه‌گیری

### پروژه ۱

- یک سیستم RAG کامل که محتوای وبسایت را خزش کرده و به سوالات پاسخ می‌دهد
- از FAISS برای جستجوی برداری سریع استفاده می‌کند
- پاسخ‌ها بر اساس محتوای واقعی وبسایت تولید می‌شوند
- منابع در پاسخ ذکر می‌شوند

### پروژه ۲

- یک سیستم ایجنت هوشمند که جایگزین اپراتورهای انسانی می‌شود
- از LangChain برای مدیریت ابزارها و ایجنت استفاده می‌کند
- ایجنت خودکار API مناسب را انتخاب و فراخوانی می‌کند
- پاسخ‌ها به زبان کاربر تولید می‌شوند

### مزایای استفاده از هوش مصنوعی

1. **سرعت بالاتر**: پاسخ‌گویی فوری به کاربران
2. **دقت بیشتر**: کاهش خطاهای انسانی
3. **هزینه کمتر**: نیاز به اپراتور کمتر
4. ** مقیاس‌پذیری**: امکان پاسخ‌گویی به تعداد زیادی کاربر همزمان
5. **پشتیبانی ۲۴/۷**: بدون نیاز به استراحت اپراتور

---

*تاریخ تهیه مستندات: ۱۴۰۵/۰۵/۰۲*
*تهیه کننده: دانشجو*
