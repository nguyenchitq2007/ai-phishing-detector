import re
from datetime import datetime
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Phishing Detector", page_icon="🛡️", layout="wide")

# ============ MODEL (cached, with graceful fallback) ============
@st.cache_resource
def load_model():
    try:
        return joblib.load("phishing_model.pkl"), joblib.load("tfidf_vectorizer.pkl")
    except Exception:
        return None, None

model, vectorizer = load_model()

# ============ CONFIG ============
KEYWORDS = {
    "en": ["urgent", "urgently", "immediately", "click here", "click now", "verify",
           "verification", "password", "account suspended", "account locked", "confirm",
           "login", "security alert", "payment", "bank", "winner", "prize",
           "congratulations", "claim now", "limited time", "otp"],
    "vi": ["khẩn cấp", "ngay lập tức", "nhấp vào đây", "bấm vào đây", "xác minh", "xác thực",
           "mật khẩu", "tài khoản bị khóa", "tài khoản bị khoá", "tạm khóa", "đăng nhập",
           "cảnh báo bảo mật", "thanh toán", "ngân hàng", "trúng thưởng", "phần thưởng",
           "chúc mừng", "nhận quà", "mã otp", "cung cấp thông tin", "hết hạn"],
}
URL_KEYWORDS = ["login", "verify", "verification", "account", "password", "secure",
                "confirm", "update", "bank", "signin", "wallet"]
BAD_TLDS = (".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq", ".icu", ".work")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"}
BRANDS = {"paypal": "paypal.com", "google": "google.com", "facebook": "facebook.com",
          "apple": "apple.com", "microsoft": "microsoft.com", "amazon": "amazon.com",
          "vietcombank": "vietcombank.com.vn", "techcombank": "techcombank.com",
          "momo": "momo.vn", "zalo": "zalo.me", "shopee": "shopee.vn"}

TEXT = {
    "en": dict(lang="🌐 Language / Ngôn ngữ", title="🛡️ AI-Based Phishing Message Detection System",
               desc="Analyze suspicious emails and messages using AI, keyword rules and URL checks.",
               tab1="📩 Analyze", tab2="📂 Batch (CSV)", tab3="🕘 History",
               label="Enter your email or message:", ph="Paste an email or message here...",
               btn="🔍 Analyze Message", empty="⚠️ Please enter a message before analyzing.",
               samples="Try a sample", s_phish="Phishing sample", s_legit="Legitimate sample",
               verdict_p="🚨 PHISHING / SUSPICIOUS MESSAGE", verdict_l="✅ LOOKS LEGITIMATE",
               score="Risk Score", level="Risk Level", ai="AI phishing probability",
               kw="Keywords found", urls="URLs found", ind="🚨 Suspicious Indicators",
               nokw="No suspicious keywords.", url_h="🔗 URL Analysis", nourl="ℹ️ No URL detected.",
               url_ok="✅ No obvious suspicious URL indicators.", url_bad="⚠️ Suspicious URL indicators:",
               nomodel="⚠️ AI model files not found – using rule-based analysis only.",
               adv_p="Do not click links, do not share passwords/OTP, verify via the official website or app.",
               adv_l="No strong phishing pattern found. Stay careful with unexpected links.",
               levels=["LOW RISK", "MEDIUM RISK", "HIGH RISK", "VERY HIGH RISK"],
               up="Upload CSV with a 'text' column", dl="⬇️ Download results", hist_empty="No analyses yet.",
               clear="🗑️ Clear history", foot="AI-Based Phishing Detection | TF-IDF + Logistic Regression + Rules"),
    "vi": dict(lang="🌐 Language / Ngôn ngữ", title="🛡️ Hệ thống phát hiện tin nhắn lừa đảo bằng AI",
               desc="Phân tích email và tin nhắn đáng ngờ bằng AI, quy tắc từ khóa và kiểm tra URL.",
               tab1="📩 Phân tích", tab2="📂 Hàng loạt (CSV)", tab3="🕘 Lịch sử",
               label="Nhập email hoặc tin nhắn của bạn:", ph="Dán email hoặc tin nhắn vào đây...",
               btn="🔍 Phân tích tin nhắn", empty="⚠️ Vui lòng nhập tin nhắn trước khi phân tích.",
               samples="Thử mẫu", s_phish="Mẫu lừa đảo", s_legit="Mẫu hợp lệ",
               verdict_p="🚨 TIN NHẮN LỪA ĐẢO / ĐÁNG NGỜ", verdict_l="✅ CÓ VẺ HỢP LỆ",
               score="Điểm rủi ro", level="Mức rủi ro", ai="Xác suất lừa đảo (AI)",
               kw="Từ khóa phát hiện", urls="URL phát hiện", ind="🚨 Dấu hiệu đáng ngờ",
               nokw="Không có từ khóa đáng ngờ.", url_h="🔗 Phân tích URL", nourl="ℹ️ Không phát hiện URL.",
               url_ok="✅ Không có dấu hiệu URL đáng ngờ rõ ràng.", url_bad="⚠️ Dấu hiệu URL đáng ngờ:",
               nomodel="⚠️ Không tìm thấy file model AI – chỉ dùng phân tích theo quy tắc.",
               adv_p="Không bấm liên kết, không cung cấp mật khẩu/OTP, hãy xác minh qua website hoặc app chính thức.",
               adv_l="Chưa thấy dấu hiệu lừa đảo mạnh. Vẫn cần cẩn trọng với liên kết lạ.",
               levels=["RỦI RO THẤP", "RỦI RO TRUNG BÌNH", "RỦI RO CAO", "RỦI RO RẤT CAO"],
               up="Tải lên CSV có cột 'text'", dl="⬇️ Tải kết quả", hist_empty="Chưa có lượt phân tích nào.",
               clear="🗑️ Xóa lịch sử", foot="Phát hiện lừa đảo bằng AI | TF-IDF + Logistic Regression + Quy tắc"),
}
SAMPLES = {
    "phish": "URGENT: Your bank account is locked! Verify your password immediately at http://secure-login-bank.xyz/verify to avoid suspension.",
    "phish_vi": "Khẩn cấp: tài khoản ngân hàng của bạn bị khóa. Nhấp vào đây để xác minh và cung cấp mã OTP: http://vietcombank-xacminh.top/login",
    "legit": "Hi team, the meeting is moved to 3pm tomorrow. Please review the agenda on our shared drive. Thanks!",
}

# ============ ANALYSIS ============
URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"\']+|\b(?:bit\.ly|tinyurl\.com|t\.co|cutt\.ly|rb\.gy)/[^\s<>"\']+', re.I)

def extract_urls(text):
    return [u.rstrip(".,;:!?)]}>'\"") for u in URL_RE.findall(text)]

def find_keywords(text):
    low, found = text.lower(), []
    for kw in KEYWORDS["en"] + KEYWORDS["vi"]:
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", low):
            found.append(kw)
    return found

def normalize_leet(s):
    return s.translate(str.maketrans("01345", "oleas"))

def analyze_url(url):
    reasons = []
    full = url if re.match(r"https?://", url, re.I) else "http://" + url
    p = urlparse(full)
    host = (p.hostname or "").lower()
    if url.lower().startswith("http://"):
        reasons.append("No HTTPS / Không dùng HTTPS")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        reasons.append("IP address instead of domain / Dùng địa chỉ IP")
    if "@" in p.netloc:
        reasons.append("Contains '@' / Chứa ký tự '@'")
    if host in SHORTENERS:
        reasons.append("URL shortener / Link rút gọn")
    if "xn--" in host:
        reasons.append("Punycode domain / Tên miền punycode")
    if host.count(".") >= 4:
        reasons.append("Too many subdomains / Quá nhiều subdomain")
    if host.endswith(BAD_TLDS):
        reasons.append(f"Suspicious TLD / Đuôi miền đáng ngờ: .{host.rsplit('.', 1)[-1]}")
    for kw in URL_KEYWORDS:
        if re.search(rf"(?<![a-z]){kw}", host + p.path.lower()):
            reasons.append(f'Contains "{kw}"')
    norm = normalize_leet(host)
    for brand, official in BRANDS.items():
        if brand in norm and not (host == official or host.endswith("." + official)):
            reasons.append(f"Possible brand impersonation / Có thể giả mạo: {brand}")
            break
    return host, list(dict.fromkeys(reasons))

def ai_probability(text):
    if model is None or vectorizer is None:
        return None
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    idx = next((i for i, c in enumerate(classes) if str(c).lower() in ("1", "phishing", "spam", "true")), len(classes) - 1)
    return float(proba[idx]) * 100

def analyze(text):
    kws = find_keywords(text)
    urls = [(u, *analyze_url(u)) for u in extract_urls(text)]
    ai = ai_probability(text)
    kw_score = min(len(kws) * 7, 25)
    url_score = min(sum(len(r) for _, _, r in urls) * 8, 30)
    score = min(100, (0.65 * ai if ai is not None else 0) + kw_score + url_score) if ai is not None \
        else min(100, kw_score * 2 + url_score * 1.5)
    return dict(score=score, ai=ai, keywords=kws, urls=urls)

def level_index(score):
    return 3 if score >= 80 else 2 if score >= 60 else 1 if score >= 30 else 0

# ============ UI ============
lang = "vi" if st.sidebar.selectbox(TEXT["en"]["lang"], ["English", "Tiếng Việt"]) == "Tiếng Việt" else "en"
T = TEXT[lang]

st.sidebar.subheader(T["samples"])
if st.sidebar.button(T["s_phish"]):
    st.session_state.msg = SAMPLES["phish_vi" if lang == "vi" else "phish"]
if st.sidebar.button(T["s_legit"]):
    st.session_state.msg = SAMPLES["legit"]

st.title(T["title"])
st.markdown(T["desc"])
if model is None:
    st.warning(T["nomodel"])
st.session_state.setdefault("history", [])

tab1, tab2, tab3 = st.tabs([T["tab1"], T["tab2"], T["tab3"]])

with tab1:
    message = st.text_area(T["label"], key="msg", height=220, placeholder=T["ph"])
    if st.button(T["btn"], use_container_width=True):
        if not message.strip():
            st.warning(T["empty"])
        else:
            r = analyze(message)
            li = level_index(r["score"])
            phishing = r["score"] >= 50
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"), "message": message[:80],
                "score": round(r["score"]), "level": T["levels"][li]})
            st.divider()
            (st.error if phishing else st.success)(T["verdict_p"] if phishing else T["verdict_l"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T["score"], f"{r['score']:.0f}/100")
            c2.metric(T["level"], T["levels"][li])
            c3.metric(T["ai"], f"{r['ai']:.1f}%" if r["ai"] is not None else "N/A")
            c4.metric(T["urls"], len(r["urls"]))
            st.progress(int(r["score"]))
            (st.warning if phishing else st.info)(T["adv_p"] if phishing else T["adv_l"])

            st.subheader(T["ind"])
            if r["keywords"]:
                st.write(", ".join(f"`{k}`" for k in r["keywords"]))
            else:
                st.info(T["nokw"])

            st.subheader(T["url_h"])
            if not r["urls"]:
                st.info(T["nourl"])
            for url, host, reasons in r["urls"]:
                st.write(f"🌐 **{url}**  \nDomain: `{host}`")
                if reasons:
                    st.warning(T["url_bad"] + "\n" + "\n".join(f"- {x}" for x in reasons))
                else:
                    st.success(T["url_ok"])

with tab2:
    file = st.file_uploader(T["up"], type="csv")
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("Missing 'text' column.")
        else:
            res = [analyze(str(t)) for t in df["text"]]
            df["risk_score"] = [round(x["score"]) for x in res]
            df["risk_level"] = [T["levels"][level_index(x["score"])] for x in res]
            df["suspicious"] = [x["score"] >= 50 for x in res]
            st.dataframe(df, use_container_width=True)
            st.download_button(T["dl"], df.to_csv(index=False).encode("utf-8-sig"), "results.csv", "text/csv")

with tab3:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button(T["clear"]):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(T["hist_empty"])

st.divider()
st.caption(T["foot"])
