# LOOKBOOK AI — Global Style Intelligence (Streamlit + Gemini)

LOOKBOOK AI is a Streamlit web app that turns an uploaded outfit photo into a **Style DNA** profile (vibe label, ideal fit/cut, color palette, key items), then translates that style to a selected country using live fashion trends, celebrity inspiration, and curated shopping links.

---

## Features

- Outfit image analysis using **Google Gemini**
- Style DNA extraction as structured JSON
- Country-specific fashion trends (Google News RSS)
- Celebrity inspiration by country & gender
- Retailer shopping links per region
- Gemini API key rotation to avoid rate limits
- Streamlit caching for performance

---

## Tech Stack

- Python 3.9+
- Streamlit
- google-generativeai (Gemini)
- Pillow
- feedparser
- requests
- python-dotenv

---

## Project Structure

```
.
├── app_full_v4.py
├── .env
└── requirements.txt
```

---

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

Windows:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install Dependencies

```
pip install streamlit google-generativeai pillow feedparser requests python-dotenv
```

---

### 3. Configure Environment Variables

Create a `.env` file in the project root.

Single key:
```
GEMINI_API_KEY=your_api_key_here
```

Multiple keys (recommended):
```
GEMINI_API_KEYS=key1,key2,key3
```

---

## Running the App

```
streamlit run app_full_v4.py
```

Open browser at:
```
http://localhost:8501
```

---

## How It Works (Technical)

1. User uploads an outfit image
2. Image is analyzed by Gemini to extract Style DNA (JSON)
3. Google News RSS fetches fashion trends by country
4. Gemini generates a localized lookbook:
   - Trend summary
   - Outfit recommendations
   - Celebrity inspiration
   - Shopping keywords
5. Retailer links are generated dynamically per country

---

## Caching

- Celebrity images cached for 24 hours
- Trend context cached for 1 hour
- Reduces API calls and improves performance

---

## Security Notes

- API keys are loaded via `.env`
- Do **NOT** commit `.env` to GitHub
- Add `.env` to `.gitignore`

---

## License

Add a LICENSE file (MIT / Apache-2.0 recommended)
