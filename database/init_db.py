import sqlite3
import bcrypt

DB = "data/bridgebot.db"

conn = sqlite3.connect(DB)

cursor = conn.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS users(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE,
                       password_hash TEXT,
                       email TEXT,
                       role TEXT,
                       status INTEGER,
                       created_date TEXT,
                       last_login TEXT
               )
               """)

password = bcrypt.hashpw(
    "admin123".encode(),
    bcrypt.gensalt()
).decode()

cursor.execute("""
               INSERT OR IGNORE INTO users
( username, password_hash, email, role, status, created_date )
VALUES ( ?, ?, ?, ?, ?, datetime('now') )
               """,
               ("admin", password, "admin@bridgebot.com", "ADMIN", 1)
               )

conn.commit()

conn.close()

print("Database Initialized")