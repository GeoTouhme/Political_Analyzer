import os
import sqlite3
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_API_KEY")
DATABASE_ID = "87aeafb55c8a4d0998efb3b186d33403"
SQLITE_DB_PATH = "/home/ubuntu/.openclaw/workspace/political_analyzer/notion_backup.sqlite"
LOG_FILE_PATH = "/home/ubuntu/.openclaw/workspace/political_analyzer/backup.log"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("backup")

def init_db():
    """Initialize the local SQLite database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            notion_id TEXT PRIMARY KEY,
            title TEXT,
            date TEXT,
            url TEXT,
            full_text TEXT,
            last_edited_time TEXT,
            backed_up_at TEXT
        )
    """)
    conn.commit()
    return conn

def fetch_all_notion_pages():
    """Generator to fetch all pages from the Notion DB."""
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    
    payload = {"page_size": 100}
    has_more = True
    next_cursor = None

    while has_more:
        if next_cursor:
            payload["start_cursor"] = next_cursor
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            log.error(f"Notion API error: {response.text}")
            break
            
        data = response.json()
        yield from data.get("results", [])
        
        has_more = data.get("has_more")
        next_cursor = data.get("next_cursor")

def backup():
    if not NOTION_TOKEN:
        log.error("NOTION_API_KEY not found in .env")
        return

    log.info("Starting Notion to SQLite backup...")
    conn = init_db()
    cursor = conn.cursor()
    
    count = 0
    for page in fetch_all_notion_pages():
        notion_id = page["id"]
        props = page["properties"]
        
        # Extract fields (mapping to your Notion schema)
        title = ""
        title_obj = props.get("Title", {}).get("title", [])
        if title_obj:
            title = title_obj[0].get("plain_text", "")
            
        date_obj = props.get("Date", {}).get("date")
        date = date_obj.get("start", "") if date_obj else ""
        url = props.get("Sources", {}).get("url", "")
        
        # Full text extraction
        text_blocks = props.get("Full text", {}).get("rich_text", [])
        full_text = " ".join([b.get("plain_text", "") for b in text_blocks])
        
        last_edited = page.get("last_edited_time", "")
        
        cursor.execute("""
            INSERT INTO articles (notion_id, title, date, url, full_text, last_edited_time, backed_up_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notion_id) DO UPDATE SET
                title=excluded.title,
                date=excluded.date,
                url=excluded.url,
                full_text=excluded.full_text,
                last_edited_time=excluded.last_edited_time,
                backed_up_at=excluded.backed_up_at
        """, (notion_id, title, date, url, full_text, last_edited, datetime.now().isoformat()))
        
        count += 1
        if count % 50 == 0:
            log.info(f"Processed {count} articles...")

    conn.commit()
    conn.close()
    log.info(f"Backup complete. Total articles in local DB: {count}")

if __name__ == "__main__":
    backup()
