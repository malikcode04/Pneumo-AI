import sqlite3
import bcrypt
from .db import DB_PATH, init_db

def create_user(username, password):
    """Create a new user."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Hash pw
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_credentials(username, password):
    """Verify login."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    
    if row:
        stored_hash = row[0].encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    return False

def init_auth():
    """Ensure DB exists and create default admin if missing."""
    init_db()
    if not check_credentials("admin", "admin"):
        create_user("admin", "admin")
