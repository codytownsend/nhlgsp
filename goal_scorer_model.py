import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from config import (
    MODEL_DIR, MODEL_FILENAME, MIN_SAMPLES_FOR_MODEL, MIN_GAMES_PER_PLAYER,
    FEATURE_COLUMNS, HOME_ADVANTAGE_FACTOR, AWAY_DISADVANTAGE_FACTOR, MAX_PROBABILITY
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("goal_scorer_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoalScorerModel:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        self.feature_cols = FEATURE_COLUMNS
        
        # Ensure model directory exists
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        # Try to load existing model
        self._load_model()
        
    def _save_model(self):
        """Save model to disk"""
        try:
            model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
            
    def _load_model(self):
        """Load model from disk if available"""
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
        
    def prepare_training_data(self, min_games=MIN_GAMES_PER_PLAYER):
        """Extract and prepare features from database for model training"""
        logger.info("Preparing training data")
        
        query = """
        WITH player_game_history AS (
            SELECT
                pgs.player_id,
                pgs.game_id,
                p.name,
                p.position,
                g.game_date,
                pgs.team_id,
                pgs.scored_goal,
                pgs.goals,
                pgs.assists,
                pgs.shots,
                pgs.time_on_ice,
                pgs.pp_goals,
                pgs.sh_goals,
                (SELECT COUNT(*) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_games,
                (SELECT SUM(scored_goal) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_goals,
                (SELECT SUM(shots) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_shots,
                -- Last 10 games stats
                (SELECT SUM(scored_goal) FROM player_game_stats pgs2
                 JOIN games g2 ON pgs2.game_id = g2.game_id
                 WHERE pgs2.player_id = pgs.player_id 
                 AND g2.game_date < g.game_date
                 ORDER BY g2.game_date DESC LIMIT 10) as last_10_goals,
                (SELECT SUM(shots) FROM player_game_stats pgs2
                 JOIN games g2 ON pgs2.game_id = g2.game_id
                 WHERE pgs2.player_id = pgs.player_id 
                 AND g2.game_date < g.game_date
                 ORDER BY g2.game_date DESC LIMIT 10) as last_10_shots,
                -- Home/Away context
                CASE WHEN g.home_team_id = pgs.team_id THEN 1 ELSE 0 END as is_home
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.status = 'Final'
        )
        SELECT
            pgh.*,
            CASE WHEN pgh.career_games >= 10 THEN pgh.career_goals * 1.0 / pgh.career_games ELSE NULL END as career_gpg,
            CASE WHEN pgh.career_shots > 0 THEN pgh.career_goals * 1.0 / pgh.career_shots ELSE NULL END as career_sh_pct,
            CASE WHEN pgh.last_10_shots > 0 THEN pgh.last_10_goals * 1.0 / pgh.last_10_shots ELSE NULL END as recent_sh_pct,
            CASE WHEN pgh.position IN ('LW', 'RW', 'C') THEN 1 ELSE 0 END as is_forward,
            CASE WHEN pgh.position = 'D' THEN 1 ELSE 0 END as is_defense,
            CASE
                WHEN pgh.career_games < ?
                THEN 0  -- Not enough data
                ELSE 1  -- Enough data for reliable calculation
            END as has_history
        FROM player_game_history pgh
        WHERE pgh.time_on_ice > 0  -- Player actually played
        """
        
        try:
            df = pd.read_sql(query, self.db_connection, params=(min_games,))
            
            # Handle missing values
            df.fillna({
                'career_gpg': 0,
                'career_sh_pct': 0,
                'recent_sh_pct': 0,
                'last_10_goals': 0,
                'last_10_shots': 0
            }, inplace=True)
            
            # Create additional features
            df['time_on_ice_minutes'] = df['time_on_ice'] / 60
            
            logger.info(f"Prepared training data with {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def train(self, min_games=MIN_GAMES_PER_PLAYER, force_retrain=False):
        """Train the model on historical data"""
        logger.info("Training model")
        
        # Skip training if model already exists and force_retrain is False
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path) and not force_retrain:
            logger.info("Model already exists. Skipping training. Use force_retrain=True to override.")
            return True
        
        try:
            features_df = self.prepare_training_data(min_games)
            
            if len(features_df) < MIN_SAMPLES_FOR_MODEL:
                logger.warning(f"Not enough data to train a robust model: only {len(features_df)} samples available")
                self.model = None
                return False
            
            # Use specified feature columns
            X = features_df[self.feature_cols]
            y = features_df['scored_goal'].astype(int)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Get probabilities for more detailed metrics
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            brier = brier_score_loss(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"Model trained on {len(X_train)} samples")
            logger.info(f"Train accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"Brier score: {brier:.4f}")
            
            # Save the model
            self._save_model()
            
            # Store evaluation metrics in database
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO model_evaluation (training_date, samples, train_accuracy, test_accuracy, "
                "roc_auc, brier_score) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), len(X_train), train_score, test_score, roc_auc, brier)
            )
            
            self.db_connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def prepare_player_features(self, player_id, game_id=None):
        """Prepare features for a single player"""
        cursor = self.db_connection.cursor()
        
        try:
            # Query for player stats
            query = """
            WITH player_stats AS (
                SELECT
                    p.player_id,
                    p.name,
                    p.position,
                    p.team_id,
                    (SELECT COUNT(*) FROM player_game_stats 
                     WHERE player_id = p.player_id) as career_games,
                    (SELECT SUM(scored_goal) FROM player_game_stats 
                     WHERE player_id = p.player_id) as career_goals,
                    (SELECT SUM(shots) FROM player_game_stats 
                     WHERE player_id = p.player_id) as career_shots,
                    (SELECT SUM(time_on_ice) FROM player_game_stats 
                     WHERE player_id = p.player_id) as career_toi,
                    (SELECT SUM(scored_goal) FROM player_game_stats pgs
                     JOIN games g ON pgs.game_id = g.game_id
                     WHERE pgs.player_id = p.player_id 
                     ORDER BY g.game_date DESC LIMIT 10) as last_10_goals,
                    (SELECT SUM(shots) FROM player_game_stats pgs
                     JOIN games g ON pgs.game_id = g.game_id
                     WHERE pgs.player_id = p.player_id 
                     ORDER BY g.game_date DESC LIMIT 10) as last_10_shots,
                    (SELECT AVG(time_on_ice) FROM player_game_stats pgs
                     JOIN games g ON pgs.game_id = g.game_id
                     WHERE pgs.player_id = p.player_id 
                     ORDER BY g.game_date DESC LIMIT 10) as last_10_avg_toi
                FROM players p
                WHERE p.player_id = ?
            )
            SELECT
                ps.*,
                CASE WHEN ps.career_games > 0 THEN ps.career_goals * 1.0 / ps.career_games ELSE 0 END as career_gpg,
                CASE WHEN ps.career_shots > 0 THEN ps.career_goals * 1.0 / ps.career_shots ELSE 0 END as career_sh_pct,
                CASE WHEN ps.last_10_shots > 0 THEN ps.last_10_goals * 1.0 / ps.last_10_shots ELSE 0 END as recent_sh_pct,
                CASE WHEN ps.career_games > 0 THEN ps.career_toi * 1.0 / (ps.career_games * 60) ELSE 0 END as avg_toi_minutes,
                CASE WHEN ps.position IN ('LW', 'RW', 'C') THEN 1 ELSE 0 END as is_forward,
                CASE WHEN ps.position = 'D' THEN 1 ELSE 0 END as is_defense,
                CASE
                    WHEN ps.career_games < 10
                    THEN 0  -- Not enough data
                    ELSE 1  -- Enough data for reliable calculation
                END as has_history
            FROM player_stats ps
            """
            
            df = pd.read_sql(query, self.db_connection, params=(player_id,))
            
            if df.empty:
                logger.warning(f"No data found for player {player_id}")
                return None
            
            # Convert time on ice if available
            if 'last_10_avg_toi' in df.columns:
                df['time_on_ice_minutes'] = df['last_10_avg_toi'] / 60
            else:
                df['time_on_ice_minutes'] = df['avg_toi_minutes']
            
            # Handle missing values
            df.fillna(0, inplace=True)
            
            return df.iloc[0]
        except Exception as e:
            logger.error(f"Error preparing features for player {player_id}: {e}")
            return None
    
    def predict_game(self, game_id):
        """Predict goal scorers for a specific game"""
        logger.info(f"Making predictions for game {game_id}")
        cursor = self.db_connection.cursor()
        
        try:
            # Get game details
            cursor.execute(
                "SELECT game_id, home_team_id, away_team_id, game_date, status FROM games WHERE game_id = ?", 
                (game_id,)
            )
            game_info = cursor.fetchone()
            
            if not game_info:
                logger.warning(f"Game ID {game_id} not found in database")
                return []
            
            game_id, home_team_id, away_team_id, game_date, status = game_info
            
            # Skip if the game is already completed
            if status == 'Final':
                logger.info(f"Game {game_id} is already completed, skipping predictions")
                return []
            
            # Get players from both teams
            query = """
            SELECT p.player_id, p.name, p.position, p.team_id, t.name as team_name
            FROM players p
            JOIN teams t ON p.team_id = t.team_id
            WHERE p.team_id IN (?, ?) AND p.active = 1
            """
            cursor.execute(query, (home_team_id, away_team_id))
            players = cursor.fetchall()
            
            # Check if we have a trained model
            if self.model is None and not self._load_model():
                logger.warning("No trained model available. Using basic probability model.")
            
            predictions = []
            for player_id, name, position, team_id, team_name in players:
                # Skip goalies
                if position == 'G':
                    continue
                
                # Get player features
                player_features = self.prepare_player_features(player_id, game_id)
                
                if player_features is None:
                    # Skip players with no features
                    continue
                
                # Determine probability
                try:
                    if self.model is not None and self.feature_cols is not None:
                        # Use ML model
                        features = player_features[self.feature_cols].values.reshape(1, -1)
                        probability = float(self.model.predict_proba(features)[0][1])
                    else:
                        # Simple probability model for small datasets
                        probability = float(player_features['career_gpg'])
                except Exception as e:
                    logger.warning(f"Error predicting for player {player_id}: {e}")
                    # Fall back to simple probability
                    probability = float(player_features['career_gpg'])
                
                # Adjust probability based on team matchup 
                if team_id == home_team_id:
                    probability *= HOME_ADVANTAGE_FACTOR
                else:
                    probability *= AWAY_DISADVANTAGE_FACTOR
                
                # Cap at maximum probability
                probability = min(MAX_PROBABILITY, probability)
                
                # Store prediction
                cursor.execute(
                    "INSERT INTO predictions (game_id, player_id, goal_probability, prediction_date, scored, last_updated) "
                    "VALUES (?, ?, ?, ?, NULL, ?)",
                    (game_id, player_id, probability, datetime.now(), datetime.now())
                )
                
                predictions.append({
                    'player_id': player_id,
                    'name': name,
                    'team': team_name,
                    'position': position,
                    'probability': probability,
                    'features': {
                        'gpg': float(player_features['career_gpg']),
                        'sh_pct': float(player_features['career_sh_pct']),
                        'games': int(player_features['career_games'])
                    }
                })
            
            self.db_connection.commit()
            
            # Sort by probability, highest first
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            logger.info(f"Generated {len(predictions)} predictions for game {game_id}")
            return predictions
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error predicting for game {game_id}: {e}")
            return []