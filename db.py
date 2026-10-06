import sqlite3
import pandas as pd

DB_PATH = "predictions.db"

def init_db():
    """Initializes the SQLite database schema."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS prediction_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      area REAL, bhk INTEGER, prop_type TEXT, region TEXT, 
                      status TEXT, age TEXT, predicted_price REAL)''')
        conn.commit()

def save_prediction(area: float, bhk: int, prop_type: str, region: str, status: str, age: str, price: float):
    """Persists a new prediction to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO prediction_history 
                     (area, bhk, prop_type, region, status, age, predicted_price)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (area, bhk, prop_type, region, status, age, price))
        conn.commit()

def load_history() -> pd.DataFrame:
    """Loads prediction history ordered by most recent."""
    with sqlite3.connect(DB_PATH) as conn:
        df_hist = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
    return df_hist
