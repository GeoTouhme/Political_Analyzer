import sqlite3
import time
from deep_translator import GoogleTranslator
import os

def translate_db(db_path, id_col):
    print(f"--- Processing {db_path} ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Add column if missing
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN translated_title TEXT")
    except: pass
    
    cursor.execute(f"SELECT {id_col}, title FROM articles WHERE translated_title IS NULL")
    rows = cursor.fetchall()
    
    translator = GoogleTranslator(source='auto', target='en')
    count = 0
    for art_id, title in rows:
        if not title: continue
        try:
            # We translate everything to ensure consistency, 
            # Google is smart enough not to change English much.
            translated = translator.translate(title)
            cursor.execute(f"UPDATE articles SET translated_title = ? WHERE {id_col} = ?", (translated, art_id))
            count += 1
            if count % 10 == 0:
                print(f"  {db_path}: Translated {count}/{len(rows)}")
                conn.commit()
            time.sleep(0.2)
        except Exception as e:
            print(f"Error: {e}")
            break
    conn.commit()
    conn.close()

if os.path.exists('political_analyzer/war_backtest_2025.sqlite'):
    translate_db('political_analyzer/war_backtest_2025.sqlite', 'id')
if os.path.exists('political_analyzer/notion_backup.sqlite'):
    translate_db('political_analyzer/notion_backup.sqlite', 'notion_id')
