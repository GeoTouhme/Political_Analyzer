import sqlite3
import time
import random
import requests
import trafilatura
from datetime import datetime

DB_PATH = 'political_analyzer/notion_backup.sqlite'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
]

def get_full_text(url):
    try:
        # Resolve Google News redirection if necessary
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        
        final_url = response.url
        print(f"    -> Final URL: {final_url[:50]}...")

        # Extract content from the final landing page
        content = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        return content
    except Exception as e:
        print(f"    [ERROR] {e}")
    return None

def process_missing_content():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Target only top 50 high priority for now to test the logic
    cursor.execute("""
        SELECT notion_id, url, title 
        FROM articles 
        WHERE (full_text IS NULL OR length(full_text) < 500)
        AND date >= '2026-02-27'
        ORDER BY date DESC
        LIMIT 50
    """)
    
    targets = cursor.fetchall()
    total = len(targets)
    print(f"--- FULL TEXT EXTRACTION (V3 - REDIRECT MODE) ---")
    
    success_count = 0
    for i, (notion_id, url, title) in enumerate(targets):
        print(f"[{i+1}/{total}] Fetching: {title[:50]}...")
        full_text = get_full_text(url)
        
        if full_text and len(full_text) > 400:
            cursor.execute("UPDATE articles SET full_text = ? WHERE notion_id = ?", (full_text, notion_id))
            conn.commit()
            success_count += 1
            print(f"    [SUCCESS] {len(full_text)} chars.")
        else:
            print(f"    [SKIP] Failed extraction.")
        
        time.sleep(random.uniform(4.0, 7.0))
    
    conn.close()
    print(f"Enriched {success_count} articles.")

if __name__ == "__main__":
    process_missing_content()
