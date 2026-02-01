import sqlite3
from pathlib import Path
import datetime
import os
import shutil

DB_PATH = Path(__file__).parent.parent.parent / "data" / "pneumoai.db"
MEDIA_ROOT = Path(__file__).parent.parent.parent / "data" / "patients"

def init_db():
    """Initialize the database tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  password_hash TEXT NOT NULL, 
                  role TEXT DEFAULT 'doctor')''')
    
    # Scans Table
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  patient_id TEXT, 
                  patient_name TEXT, 
                  scan_date TEXT, 
                  diagnosis TEXT, 
                  confidence REAL, 
                  image_path TEXT,
                  heatmap_path TEXT,
                  notes TEXT,
                  doctor_username TEXT)''')
    
    # Migration: Add doctor_username column if it doesn't exist
    try:
        c.execute("ALTER TABLE scans ADD COLUMN doctor_username TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    conn.commit()
    conn.close()
    
def add_scan(patient_id, patient_name, diagnosis, confidence, original_image, heatmap_image, doctor_username, notes=""):
    """
    Saves scan metadata to DB and images to disk.
    Returns the scan ID.
    """
    # 1. Save Images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = MEDIA_ROOT / patient_id / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = save_dir / "original.jpg"
    heat_path = save_dir / "heatmap.jpg"
    
    original_image.save(img_path)
    heatmap_image.save(heat_path)
    
    # 2. Save DB Record
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''INSERT INTO scans 
                 (patient_id, patient_name, scan_date, diagnosis, confidence, image_path, heatmap_path, doctor_username, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (patient_id, patient_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               diagnosis, float(confidence), str(img_path), str(heat_path), doctor_username, notes))
    conn.commit()
    scan_id = c.lastrowid
    conn.close()
    return scan_id

def get_all_scans(doctor_username=None):
    """Retrieve all scans, optionally filtered by doctor."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if doctor_username:
        c.execute("SELECT * FROM scans WHERE doctor_username = ? ORDER BY scan_date DESC", (doctor_username,))
    else:
        c.execute("SELECT * FROM scans ORDER BY scan_date DESC")
        
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def delete_scan(scan_id):
    """Delete a scan record and its files."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get paths to delete files
    c.execute("SELECT image_path, heatmap_path FROM scans WHERE id=?", (scan_id,))
    row = c.fetchone()
    if row:
        try:
            # Delete directory (parent of the image)
            folder = Path(row['image_path']).parent
            if folder.exists():
                shutil.rmtree(folder)
        except Exception as e:
            print(f"Error deleting files: {e}")
            
    c.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    conn.commit()
    conn.close()

def update_scan(scan_id, new_name, new_notes):
    """Update patient name and notes for a scan."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''UPDATE scans 
                 SET patient_name = ?, notes = ?
                 WHERE id = ?''', (new_name, new_notes, scan_id))
    conn.commit()
    conn.close()
