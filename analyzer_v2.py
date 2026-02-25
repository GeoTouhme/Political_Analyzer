import os
import requests
import json
from textblob import TextBlob
from datetime import datetime

# Configuration (Loads .env)
def load_env_var(key, default=None):
    val = os.environ.get(key)
    if val: return val
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.split('=', 1)[1].strip()
    return default

NOTION_TOKEN = load_env_var("NOTION_API_KEY")
DATA_SOURCE_ID = "a9327cf7-8083-433e-aa7e-bca30160ffb6"

# --- 1. Strategic Dictionary (Risk Scoring) ---
MILITARY_TERMS = {
    'critical': ['decapitation', 'regime change', 'invasion', 'existential', 'pre-emptive', 'nuclear breakout', 'massive retaliation'],
    'high': ['strike', 'carrier', 'armada', 'missile', 'buildup', 'offensive doctrine', 'punitive', 'escalation'],
    'medium': ['deployment', 'assets', 'patrol', 'exercise', 'maneuver', 'posture']
}

DIPLOMATIC_TERMS = {
    'high': ['treaty', 'agreement', 'breakthrough', 'rapprochement', 'normalization'],
    'medium': ['talks', 'negotiation', 'dialogue', 'concession', 'relief'],
    'low': ['meeting', 'statement', 'visit', 'consultation']
}

def fetch_notion_data():
    url = f"https://api.notion.com/v1/data_sources/{DATA_SOURCE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching: {response.text}")
        return []
    return response.json().get("results", [])

def calculate_risk_score(text, sentiment):
    # 1. Base Score from Sentiment (Negative = Risky)
    # Range: -1.0 (Very Negative) to 1.0 (Very Positive)
    # Map: -1.0 -> +40 Risk Points, 1.0 -> -10 Risk Points
    base_score = (1 - sentiment) * 20 

    term_score = 0
    text_lower = text.lower()
    
    # 2. Military Terms Weighting (Adds Risk)
    for term in MILITARY_TERMS['critical']:
        if term in text_lower: term_score += 15
    for term in MILITARY_TERMS['high']:
        if term in text_lower: term_score += 8
    for term in MILITARY_TERMS['medium']:
        if term in text_lower: term_score += 3
        
    # 3. Diplomatic Terms Mitigation (Reduces Risk)
    for term in DIPLOMATIC_TERMS['high']:
        if term in text_lower: term_score -= 10
    for term in DIPLOMATIC_TERMS['medium']:
        if term in text_lower: term_score -= 5
    
    total_risk = base_score + term_score
    return max(0, min(100, total_risk)) # Clamp 0-100

def analyze_article(article):
    props = article.get("properties", {})
    
    # Safe Extraction
    title_list = props.get("Title", {}).get("title", [])
    title = title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"
    
    full_text_list = props.get("Full text", {}).get("rich_text", [])
    full_text = full_text_list[0].get("plain_text", "") if full_text_list else ""
    
    if not full_text: return None
        
    # NLP Processing
    blob = TextBlob(full_text)
    sentiment = blob.sentiment.polarity
    risk_score = calculate_risk_score(full_text, sentiment)
    
    # Categorize Risk Level
    if risk_score > 75: level = "CRITICAL"
    elif risk_score > 50: level = "HIGH"
    elif risk_score > 25: level = "MEDIUM"
    else: level = "LOW"
    
    return {
        "title": title,
        "sentiment_polarity": round(sentiment, 2),
        "risk_score": round(risk_score, 1),
        "risk_level": level,
        "key_phrases": blob.noun_phrases[:5] # Top 5 Noun Phrases
    }

def main():
    print("--- Political Pattern Analyzer v0.2 ---")
    articles = fetch_notion_data()
    print(f"Analyzing {len(articles)} articles from Notion...\n")
    
    results = []
    for article in articles:
        analysis = analyze_article(article)
        if analysis:
            results.append(analysis)
            print(f"[{analysis['risk_level']}] {analysis['title']}")
            print(f"  Risk Score: {analysis['risk_score']} / 100")
            print(f"  Sentiment: {analysis['sentiment_polarity']} (Neg < 0 < Pos)")
            print(f"  Keywords: {', '.join(analysis['key_phrases'])}")
            print("-" * 50)
            
    # Save Report
    with open('analysis_report_v2.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n✅ Report exported to 'analysis_report_v2.json'")

if __name__ == "__main__":
    main()