"""
migrate_db.py
─────────────
Run this ONCE to add the three new columns to your existing
pneumoinsight.db database. It will not delete any existing records.

Usage:
    python migrate_db.py

Run this from the same folder as your app.py (where pneumoinsight.db lives).
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'instance', 'pneumoinsight.db')

# Fallback if db is in current directory
if not os.path.exists(DB_PATH):
    DB_PATH = 'pneumoinsight.db'
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join('instance', 'pneumoinsight.db')

print(f"Using database at: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(patient_record)")
existing_cols = [row[1] for row in cursor.fetchall()]
print(f"Existing columns: {existing_cols}")

# Add new columns only if they don't already exist
new_columns = [
    ("heatmap_raw_filename", "VARCHAR(100)"),
    ("explanation",          "TEXT"),
    ("confidence_tier",      "VARCHAR(30)"),
]

for col_name, col_type in new_columns:
    if col_name not in existing_cols:
        cursor.execute(f"ALTER TABLE patient_record ADD COLUMN {col_name} {col_type}")
        print(f"  + Added column: {col_name}")
    else:
        print(f"  ✓ Column already exists: {col_name}")

conn.commit()
conn.close()
print("\nMigration complete. Your existing records are untouched.")
print("New records created after this migration will include full explainability data.")
