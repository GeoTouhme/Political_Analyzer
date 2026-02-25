import os
import requests
import json

# Configuration
def load_env_var(key, default=None):
    # Try getting from OS environment first
    val = os.environ.get(key)
    if val:
        return val
    
    # Try reading from .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.split('=', 1)[1].strip()
    return default

NOTION_TOKEN = load_env_var("NOTION_API_KEY")
DATA_SOURCE_ID = "a9327cf7-8083-433e-aa7e-bca30160ffb6" # The id for 'Political Analyses' Data Source

def fetch_notion_data():
    url = f"https://api.notion.com/v1/data_sources/{DATA_SOURCE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.text}")
        return []
    
    return response.json().get("results", [])

def analyze_article(article):
    properties = article.get("properties", {})
    title_list = properties.get("Title", {}).get("title", [])
    title = title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"
    
    full_text_list = properties.get("Full text", {}).get("rich_text", [])
    full_text = full_text_list[0].get("plain_text", "") if full_text_list else ""
    
    # Placeholder for NLP logic (VADER/spaCy)
    # For now, we extract keyword presence as a basic pattern
    keywords = ["offensive", "existential", "deterrence", "escalation", "strike"]
    found_keywords = [word for word in keywords if word.lower() in full_text.lower()]
    
    return {
        "title": title,
        "patterns": found_keywords,
        "text_length": len(full_text)
    }

def main():
    print(f"--- Political Pattern Analyzer v0.1 ---")
    articles = fetch_notion_data()
    print(f"Found {len(articles)} articles in Notion.\n")
    
    for article in articles:
        analysis = analyze_article(article)
        print(f"Article: {analysis['title']}")
        print(f"Detected Patterns: {', '.join(analysis['patterns']) if analysis['patterns'] else 'None'}")
        print(f"Content Depth: {analysis['text_length']} chars")
        print("-" * 30)

if __name__ == "__main__":
    main()
