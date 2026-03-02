import feedparser
import sqlite3
import hashlib
from datetime import datetime
import time
import urllib.parse

# Comprehensive RSS list for full scale war monitoring
CORE_FEEDS = {
    'Al Jazeera English': 'https://www.aljazeera.com/xml/rss/all.xml',
    'Reuters World': 'https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US&gl=US&ceid=US:en',
    'BBC World': 'https://feeds.bbci.co.uk/news/world/rss.xml',
    'IRNA English': 'https://en.irna.ir/rss',
    'Times of Israel': 'https://www.timesofisrael.com/feed/',
    'Guardian Middle East': 'https://www.theguardian.com/world/middleeast/rss',
    'Defense News': 'https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml',
    'USNI News': 'https://news.usni.org/feed',
    'The Aviationist': 'https://theaviationist.com/feed/',
    'Jerusalem Post': 'https://news.google.com/rss/search?q=site:jpost.com&hl=en-US&gl=US&ceid=US:en',
    'Tehran Times': 'https://news.google.com/rss/search?q=site:tehrantimes.com&hl=en-US&gl=US&ceid=US:en'
}

# Google News RSS Queries (The "Secret Sauce")
QUERIES = [
    "Iran US Israel conflict",
    "Pentagon Iran military action",
    "Middle East oil supply disruption",
    "Strait of Hormuz tension",
    "Red Sea naval battle",
    "Iran supreme leader news",
    "Russia China response Iran attack",
    "Lebanon border escalation"
]

DB_PATH = '/home/ubuntu/.openclaw/workspace/political_analyzer/notion_backup.sqlite'

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def collect_rss():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_count = 0
    
    print(f"--- GLOBAL RSS WAR MONITOR STARTING ---")
    
    # Process Core Feeds
    for name, url in CORE_FEEDS.items():
        print(f"Polling {name}...")
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            
            cursor.execute("SELECT title FROM articles WHERE title = ? OR url = ?", (title, link))
            if cursor.fetchone(): continue
            
            timestamp = datetime.now().isoformat()
            # Try to get date from entry
            date_str = datetime.now().strftime('%Y-%m-%d')
            
            try:
                cursor.execute('''INSERT INTO articles (notion_id, title, date, url, full_text, last_edited_time, backed_up_at)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (f"rss_{get_hash(title)}", title, date_str, link, entry.get('summary', ''), timestamp, timestamp))
                new_count += 1
            except: continue

    # Process Search Queries
    for q in QUERIES:
        print(f"Searching for: {q}...")
        encoded_q = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={encoded_q}+when:2d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            cursor.execute("SELECT title FROM articles WHERE title = ? OR url = ?", (title, link))
            if cursor.fetchone(): continue
            
            timestamp = datetime.now().isoformat()
            try:
                cursor.execute('''INSERT INTO articles (notion_id, title, date, url, full_text, last_edited_time, backed_up_at)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (f"q_{get_hash(title)}", title, datetime.now().strftime('%Y-%m-%d'), link, entry.get('summary', ''), timestamp, timestamp))
                new_count += 1
            except: continue
            
    conn.commit()
    conn.close()
    print(f"--- MISSION COMPLETE: Added {new_count} NEW articles to Situation Room ---")

if __name__ == "__main__":
    collect_rss()
