import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_API_KEY", "")
NOTION_DB_ID = "87aeafb55c8a4d0998efb3b186d33403"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("syncer")

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

def fetch_existing_titles():
    url = f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query"
    existing = set()
    payload = {"page_size": 100}
    while True:
        resp = requests.post(url, headers=notion_headers(), json=payload)
        if resp.status_code != 200: break
        data = resp.json()
        for page in data.get("results", []):
            props = page.get("properties", {})
            title_list = props.get("Title", {}).get("title", [])
            if title_list:
                existing.add(title_list[0].get("plain_text", "").lower().strip())
        if not data.get("has_more"): break
        payload["start_cursor"] = data["next_cursor"]
    return existing

def save_to_notion(date, source, title, event, region, actors):
    full_text = f"TITLE: {title}\nSOURCE: {source}\nDATE: {date}\nREGION: {region}\nPRIMARY_ACTORS: {actors}\nEVENT_TYPE: {event}\n\n(Recovered from log)"
    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": title[:2000]}}]},
            "Date": {"date": {"start": date}},
            "Full text": {"rich_text": [{"text": {"content": full_text[:2000]}}]},
        }
    }
    resp = requests.post("https://api.notion.com/v1/pages", headers=notion_headers(), json=payload)
    return resp.status_code in (200, 201)

def sync():
    existing = fetch_existing_titles()
    log.info(f"Found {len(existing)} existing articles in Notion.")
    
    with open("political_analyzer/collector.log", "r") as f:
        lines = f.readlines()
        
    matches = []
    for i in range(len(lines)):
        if "[DRY-RUN] Would save:" in lines[i]:
            parts = lines[i].split("Would save: ")
            if len(parts) < 2: continue
            data = parts[1].split(" | ")
            if len(data) < 3: continue
            
            date, source, title = data[0].strip(), data[1].strip(), data[2].strip()
            
            event, region, actors = "N/A", "N/A", "N/A"
            if i + 1 < len(lines) and "Event:" in lines[i+1]:
                meta_line = lines[i+1].split("Event: ")[1]
                meta_parts = meta_line.split(" | ")
                event = meta_parts[0].strip()
                for mp in meta_parts:
                    if "Region: " in mp: region = mp.replace("Region: ", "").strip()
                    elif "Actors: " in mp: actors = mp.replace("Actors: ", "").strip()
            
            matches.append((date, source, title, event, region, actors))

    log.info(f"Parsed {len(matches)} articles from log.")
    
    saved_count = 0
    for date, source, title, event, region, actors in matches:
        if title.lower() not in existing:
            if save_to_notion(date, source, title, event, region, actors):
                saved_count += 1
                existing.add(title.lower())
                log.info(f"Saved: {title[:50]}...")
    
    log.info(f"Sync complete. Saved {saved_count} new articles.")
    log.info(f"Total articles in Notion now: {len(existing)}")

if __name__ == "__main__":
    sync()
