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
ENSEMBLE_MODEL_DIR = 'ensemble_models'
MIN_SAMPLES_FOR_MODEL = 500  
MIN_GAMES_PER_PLAYER = 20  

# Feature configuration - New expanded feature set
FEATURE_COLUMNS = [
    # Basic statistics
    'career_gpg',
    'career_sh_pct',
    'recent_sh_pct',
    'time_on_ice_minutes', 
    'career_games',
    
    # Position indicators
    'is_forward',
    'is_defense',
    'position_weight',
    
    # Rate statistics
    'goals_per_60',
    'shots_per_60',
    
    # Streak metrics
    'goals_last_3',
    'goals_last_5',
    'consecutive_games_with_goal',
    'days_since_last_goal',
    'exponentially_weighted_goals',
    'goal_momentum',
    
    # Shooting metrics
    'shooting_pct_variance',
    'expected_regression',
    'shot_quality_index',
    
    # Team context
    'goals_relative_to_team',
    'shots_relative_to_team',
    'shooting_efficiency_relative',
    
    # Game context
    'is_home',
    'opponent_strength',
    'gpg_vs_opponent',
    
    # Experience/reliability
    'has_history'
]

# Feature groups for ensemble models
FORWARDS_FEATURES = [
    'career_gpg', 'career_sh_pct', 'recent_sh_pct', 'time_on_ice_minutes',
    'goals_per_60', 'goals_last_3', 'goals_last_5', 'consecutive_games_with_goal',
    'exponentially_weighted_goals', 'shot_quality_index', 'is_home', 'opponent_strength'
]

DEFENSE_FEATURES = [
    'career_gpg', 'career_sh_pct', 'time_on_ice_minutes', 'goals_last_5',
    'shots_per_60', 'goals_per_60', 'is_home', 'opponent_strength'
]

# Prediction configuration
HOME_ADVANTAGE_FACTOR = 1.05
AWAY_DISADVANTAGE_FACTOR = 0.95
MAX_PROBABILITY = 0.40
RECENCY_WEIGHT = 0.7

# Position weights (multiplier for probability)
POSITION_WEIGHTS = {
    'C': 1.0,
    'L': 1.0,
    'R': 1.0,
    'D': 0.5,
    'G': 0.0
}

# Data collection
DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_PREDICTION_DAYS = 1

# League averages for Bayesian shrinkage
LEAGUE_AVG_SHOOTING_PCT = 0.095
LEAGUE_AVG_GOALS_PER_GAME = {
    'F': 0.16,
    'D': 0.06
}

# Sample size threshold for full weight on player stats
FULL_HISTORY_GAMES = 50

# Model evaluation parameters
CROSS_VALIDATION_FOLDS = 5
FEATURE_SELECTION_METHOD = 'recursive'  # Options: 'recursive', 'lasso', 'permutation'

# NHL Team ID to Name mapping
NHL_TEAM_NAMES = {
    1: "New Jersey Devils",
    2: "New York Islanders",
    3: "New York Rangers",
    4: "Philadelphia Flyers",
    5: "Pittsburgh Penguins",
    6: "Boston Bruins",
    7: "Buffalo Sabres",
    8: "Montreal Canadiens",
    9: "Ottawa Senators",
    10: "Toronto Maple Leafs",
    12: "Carolina Hurricanes",
    13: "Florida Panthers",
    14: "Tampa Bay Lightning",
    15: "Washington Capitals",
    16: "Chicago Blackhawks",
    17: "Detroit Red Wings",
    18: "Nashville Predators",
    19: "St. Louis Blues",
    20: "Calgary Flames",
    21: "Colorado Avalanche",
    22: "Edmonton Oilers",
    23: "Vancouver Canucks",
    24: "Anaheim Ducks",
    25: "Dallas Stars",
    26: "Los Angeles Kings",
    28: "San Jose Sharks",
    29: "Columbus Blue Jackets",
    30: "Minnesota Wild",
    52: "Winnipeg Jets",
    53: "Arizona Coyotes",
    54: "Vegas Golden Knights",
    55: "Seattle Kraken",
    # Add any news teams if needed
}