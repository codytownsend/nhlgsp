import sqlite3
from datetime import datetime

def create_database():
    conn = sqlite3.connect('nhl_predictions.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        name TEXT,
        team_id INTEGER,
        position TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS player_stats (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        season TEXT,
        games INTEGER,
        goals INTEGER,
        assists INTEGER,
        shots INTEGER,
        time_on_ice INTEGER,
        pp_goals INTEGER,
        sh_goals INTEGER,
        game_winning_goals INTEGER,
        last_updated TIMESTAMP,
        FOREIGN KEY (player_id) REFERENCES players (player_id)
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY,
        season TEXT,
        game_date DATE,
        home_team_id INTEGER,
        away_team_id INTEGER,
        status TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        player_id INTEGER,
        goal_probability REAL,
        prediction_date TIMESTAMP,
        scored BOOLEAN,
        FOREIGN KEY (game_id) REFERENCES games (game_id),
        FOREIGN KEY (player_id) REFERENCES players (player_id)
    )
    ''')
    
    conn.commit()
    return conn

def get_connection():
    """Connect to the database"""
    return sqlite3.connect('nhl_predictions.db')