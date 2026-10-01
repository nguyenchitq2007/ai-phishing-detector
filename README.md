# 🛡️ AI-Based Phishing Message Detection System

Streamlit web app (English / Tiếng Việt) that detects phishing emails and messages using
**TF-IDF + Logistic Regression**, bilingual keyword rules and URL heuristics (hybrid scoring).

## Features
- Hybrid risk score (0-100): AI probability + keywords + URL analysis
- URL checks: no HTTPS, IP host, `@`, shorteners, punycode, bad TLD, brand impersonation
- Batch analysis via CSV, downloadable results, session history
- Works rule-only if model files are missing

## Run
```bash
pip install -r requirements.txt
python train_model.py data.csv      # optional: retrain (columns: text,label)
streamlit run app.py
```
Keep `phishing_model.pkl` and `tfidf_vectorizer.pkl` next to `app.py`.

## Deploy (free)
Push to GitHub → https://share.streamlit.io → select repo → main file `app.py`.

## Limitations
The model only understands languages present in its training data; Vietnamese support
relies on keyword rules unless retrained with Vietnamese data.
