import requests
import sqlite3
import hashlib
import time

DB_PATH = 'political_analyzer/war_backtest_2025.sqlite'
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def fetch_june_war():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We broaden the queries as June 2025 is near the limit of Doc API's history
    queries = [
        "Iran Israel military",
        "Tehran Tel Aviv",
        "nuclear facility Iran",
        "Israel Iran conflict"
    ]
    
    total_added = 0
    # Search in two buckets: March-May and June-August 2025
    buckets = [
        ("20250301000000", "20250531235959"),
        ("20250601000000", "20250831235959")
    ]

    for start, end in buckets:
        for q in queries:
            print(f"Querying [{start[:8]}]: {q}...")
            params = {
                "query": f'"{q}" sourcelang:eng',
                "mode": "artlist",
                "maxrecords": 250,
                "format": "json",
                "startdatetime": start,
                "enddatetime": end
            }
            try:
                r = requests.get(BASE_URL, params=params, timeout=40)
                if r.status_code == 200:
                    articles = r.json().get('articles', [])
                    for a in articles:
                        try:
                            raw_date = a['seendate']
                            formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                            cursor.execute("INSERT OR IGNORE INTO articles (id, title, date, url, source) VALUES (?, ?, ?, ?, ?)",
                                         (get_hash(a['url']), a['title'], formatted_date, a['url'], a['sourcecountry']))
                            total_added += 1
                        except: continue
                    print(f"  [+] Found {len(articles)} items.")
                else:
                    print(f"  [-] API Error {r.status_code}")
            except Exception as e:
                print(f"  [-] Failed: {e}")
            time.sleep(2)
        conn.commit()

    conn.close()
    print(f"\n--- SUCCESS: Added {total_added} articles ---")

if __name__ == "__main__":
    fetch_june_war()
