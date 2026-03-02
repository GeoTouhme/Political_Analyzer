import requests
import sqlite3
import hashlib
import time

DB_PATH = 'political_analyzer/war_backtest_2025.sqlite'
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DROP TABLE IF EXISTS articles')
    conn.execute('CREATE TABLE articles (id TEXT, title TEXT, date TEXT, url TEXT, source TEXT)')
    conn.close()

def run():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    # Testing with a very simple Broad query to see if ANY data comes back
    query = "Iran Israel conflict" 
    print(f"Testing GDELT with query: {query}")
    
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 50,
        "format": "json"
    }
    
    r = requests.get(BASE_URL, params=params)
    print(f"Status: {r.status_code}")
    data = r.json()
    articles = data.get('articles', [])
    print(f"Found: {len(articles)}")
    
    for a in articles:
        conn.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?)",
                    (hashlib.md5(a['url'].encode()).hexdigest(), a['title'], a['seendate'][:8], a['url'], a['sourcecountry']))
    
    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    run()
