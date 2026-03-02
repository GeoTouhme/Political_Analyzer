import feedparser
import sqlite3
import hashlib
from datetime import datetime, timedelta
import time
import urllib.parse

# Comprehensive queries focused on the specific date window
HISTORICAL_QUERIES = [
    "Iran US Israel attack Feb 28",
    "Israel preemptive strike Iran Feb 28",
    "Tehran explosions February 28 2026",
    "Supreme Leader Ali Khamenei dead news",
    "Operation Epic Fury US Israel",
    "Iran missile retaliation Gulf Feb 28",
    "Middle East war start February 2026"
]

DB_PATH = 'political_analyzer/notion_backup.sqlite'

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def collect_historical_rss():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_count = 0
    
    print(f"--- TARGETED HISTORICAL RSS COLLECTION (Feb 27-28) ---")
    
    for q in HISTORICAL_QUERIES:
        print(f"Deep searching: {q}...")
        # We use Google News RSS with specific daterange logic
        # Note: Google News RSS when:Xh or when:Xd is more reliable than absolute dates in RSS feed,
        # but we will manually filter the results to ensure they belong to the window.
        encoded_q = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={encoded_q}+after:2026-02-26+before:2026-03-01&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            
            # Deduplicate
            cursor.execute("SELECT title FROM articles WHERE title = ? OR url = ?", (title, link))
            if cursor.fetchone(): continue
            
            # Parse date to confirm it's 27 or 28 Feb
            # Google News usually provides published in entry.published
            published_parsed = entry.get('published_parsed')
            if published_parsed:
                p_date = datetime.fromtimestamp(time.mktime(published_parsed))
                date_str = p_date.strftime('%Y-%m-%d')
                if date_str not in ['2026-02-27', '2026-02-28']:
                    continue # Skip if outside target window
            else:
                # Fallback: manually assign 2026-02-28 for these specific queries if date missing
                date_str = '2026-02-28'

            timestamp = datetime.now().isoformat()
            try:
                cursor.execute('''INSERT INTO articles (notion_id, title, date, url, full_text, last_edited_time, backed_up_at)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (f"hist_{get_hash(title)}", title, date_str, link, entry.get('summary', ''), timestamp, timestamp))
                new_count += 1
            except: continue
            
    conn.commit()
    conn.close()
    print(f"--- SUCCESS: Added {new_count} targeted historical articles (Feb 27-28) ---")

if __name__ == "__main__":
    collect_historical_rss()
