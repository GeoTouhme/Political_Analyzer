import sqlite3
import pandas as pd

conn = sqlite3.connect('political_analyzer/notion_backup.sqlite')
# Check last 3 days of data specifically
query = "SELECT date, title, source_url FROM articles WHERE date >= '2026-02-27' ORDER BY date DESC"
df = pd.read_sql_query(query, conn)

print(f"--- DATABASE CONTENT CHECK (LAST 3 DAYS) ---")
if df.empty:
    print("No articles found from Feb 27 onwards.")
else:
    print(df.to_string())

conn.close()
