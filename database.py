import sqlite3
import os
import logging
from datetime import datetime
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

def create_database():
    """Create database and tables if they don't exist"""
    # Check if database directory exists
    db_dir = os.path.dirname(DATABASE_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    # Create new connection
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    c = conn.cursor()
    
    # Teams table
    c.execute('''
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT,
        abbreviation TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    # Players table
    c.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        name TEXT,
        team_id INTEGER,
        position TEXT,
        active BOOLEAN DEFAULT 1,
        last_updated TIMESTAMP,
        FOREIGN KEY (team_id) REFERENCES teams (team_id)
    )
    ''')
    
    # Games table
    c.execute('''
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY,
        season TEXT,
        game_date DATE,
        home_team_id INTEGER,
        away_team_id INTEGER,
        status TEXT,
        home_score INTEGER,
        away_score INTEGER,
        last_updated TIMESTAMP,
        FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams (team_id)
    )
    ''')
    
    # Player season stats
    c.execute('''
    CREATE TABLE IF NOT EXISTS player_stats (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        season TEXT,
        games INTEGER,
        goals INTEGER,
        assists INTEGER,
        points INTEGER,
        shots INTEGER,
        time_on_ice INTEGER,
        pp_goals INTEGER,
        sh_goals INTEGER,
        game_winning_goals INTEGER,
        shooting_pct REAL,
        last_updated TIMESTAMP,
        FOREIGN KEY (player_id) REFERENCES players (player_id),
        UNIQUE(player_id, season)
    )
    ''')
    
    # Player game stats
    c.execute('''
    CREATE TABLE IF NOT EXISTS player_game_stats (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        game_id INTEGER,
        team_id INTEGER,
        position TEXT,
        scored_goal BOOLEAN DEFAULT 0,
        goals INTEGER DEFAULT 0,
        assists INTEGER DEFAULT 0,
        shots INTEGER DEFAULT 0,
        time_on_ice INTEGER DEFAULT 0,
        pp_goals INTEGER DEFAULT 0,
        sh_goals INTEGER DEFAULT 0,
        last_updated TIMESTAMP,
        FOREIGN KEY (player_id) REFERENCES players (player_id),
        FOREIGN KEY (game_id) REFERENCES games (game_id),
        FOREIGN KEY (team_id) REFERENCES teams (team_id),
        UNIQUE(player_id, game_id)
    )
    ''')
    
    # Predictions table
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        player_id INTEGER,
        goal_probability REAL,
        prediction_date TIMESTAMP,
        scored BOOLEAN,
        last_updated TIMESTAMP,
        FOREIGN KEY (game_id) REFERENCES games (game_id),
        FOREIGN KEY (player_id) REFERENCES players (player_id)
    )
    ''')
    
    # Model evaluation table
    c.execute('''
    CREATE TABLE IF NOT EXISTS model_evaluation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_date TIMESTAMP,
        samples INTEGER,
        train_accuracy REAL,
        test_accuracy REAL,
        roc_auc REAL,
        brier_score REAL
    )
    ''')
    
    # Prediction evaluation table
    c.execute('''
    CREATE TABLE IF NOT EXISTS prediction_evaluation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_date TIMESTAMP,
        target_date DATE,
        correct_predictions INTEGER, 
        total_predictions INTEGER,
        accuracy REAL
    )
    ''')
    
    # Create indexes for better performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_id ON player_game_stats(player_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_player_game_stats_game_id ON player_game_stats(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)')
    
    conn.commit()
    logger.info(f"Database initialized at {DATABASE_PATH}")
    
    return conn

def get_connection():
    """Connect to the database with foreign key constraints enabled"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def execute_query(query, params=(), fetch=True, many=False):
    """
    Execute a query with error handling
    
    Args:
        query (str): SQL query to execute
        params (tuple or list): Parameters for the query
        fetch (bool): Whether to fetch results
        many (bool): Whether to execute many statements
        
    Returns:
        list or None: Query results if fetch=True, else None
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if many:
            cursor.executemany(query, params)
        else:
            cursor.execute(query, params)
            
        if fetch:
            results = cursor.fetchall()
        else:
            conn.commit()
            results = None
            
        return results
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()