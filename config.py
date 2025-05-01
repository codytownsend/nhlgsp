"""
Configuration settings for the NHL Goal Scorer Prediction System
"""

# Database configuration
DATABASE_PATH = 'nhl_predictions.db'

# API configuration
NHL_WEB_API_BASE_URL = 'https://api-web.nhle.com/v1'
NHL_STATS_API_BASE_URL = 'https://api.nhle.com/stats/rest/en'
API_REQUEST_TIMEOUT = 10  # seconds
API_MAX_RETRIES = 3
API_RETRY_DELAY = 5  # seconds

# Model configuration
MODEL_DIR = 'models'
MODEL_FILENAME = 'goal_scorer_model.joblib'
MIN_SAMPLES_FOR_MODEL = 100
MIN_GAMES_PER_PLAYER = 10

# Feature configuration
FEATURE_COLUMNS = [
    'career_gpg', 
    'career_sh_pct', 
    'recent_sh_pct', 
    'is_forward', 
    'time_on_ice_minutes', 
    'has_history'
]

# Prediction configuration
HOME_ADVANTAGE_FACTOR = 1.1
AWAY_DISADVANTAGE_FACTOR = 0.9
MAX_PROBABILITY = 0.95

# Data collection
DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_PREDICTION_DAYS = 1