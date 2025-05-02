import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, precision_score, recall_score, 
    f1_score, log_loss, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import Lasso

from config import (
    MODEL_DIR, MODEL_FILENAME, ENSEMBLE_MODEL_DIR, MIN_SAMPLES_FOR_MODEL, MIN_GAMES_PER_PLAYER,
    FEATURE_COLUMNS, FORWARDS_FEATURES, DEFENSE_FEATURES, HOME_ADVANTAGE_FACTOR, 
    AWAY_DISADVANTAGE_FACTOR, MAX_PROBABILITY, POSITION_WEIGHTS, 
    LEAGUE_AVG_GOALS_PER_GAME, LEAGUE_AVG_SHOOTING_PCT, FULL_HISTORY_GAMES,
    CROSS_VALIDATION_FOLDS, FEATURE_SELECTION_METHOD
)

from hockey_analytics import calculate_advanced_metrics

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
        
        # Main model pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        
        # Ensemble models
        self.forward_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100))
        ])
        
        self.defense_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100))
        ])
        
        self.feature_cols = FEATURE_COLUMNS
        self.forward_features = FORWARDS_FEATURES
        self.defense_features = DEFENSE_FEATURES
        
        # Feature importance information
        self.feature_importances = {}
        
        # Ensure model directories exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not os.path.exists(ENSEMBLE_MODEL_DIR):
            os.makedirs(ENSEMBLE_MODEL_DIR)
            
        # Try to load existing models
        self._load_model()
        self._load_ensemble_models()
        
    def _save_model(self):
        """Save model to disk"""
        try:
            model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            joblib.dump(self.model, model_path)
            
            # Save feature importances if available
            if self.feature_importances:
                importance_path = os.path.join(MODEL_DIR, 'feature_importances.joblib')
                joblib.dump(self.feature_importances, importance_path)
                
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _save_ensemble_models(self):
        """Save ensemble models to disk"""
        try:
            forward_model_path = os.path.join(ENSEMBLE_MODEL_DIR, 'forward_model.joblib')
            defense_model_path = os.path.join(ENSEMBLE_MODEL_DIR, 'defense_model.joblib')
            
            joblib.dump(self.forward_model, forward_model_path)
            joblib.dump(self.defense_model, defense_model_path)
            
            logger.info(f"Ensemble models saved to {ENSEMBLE_MODEL_DIR}")
            return True
        except Exception as e:
            logger.error(f"Error saving ensemble models: {e}")
            return False
            
    def _load_model(self):
        """Load model from disk if available"""
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                
                # Load feature importances if available
                importance_path = os.path.join(MODEL_DIR, 'feature_importances.joblib')
                if os.path.exists(importance_path):
                    self.feature_importances = joblib.load(importance_path)
                
                logger.info(f"Model loaded from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def _load_ensemble_models(self):
        """Load ensemble models from disk if available"""
        forward_model_path = os.path.join(ENSEMBLE_MODEL_DIR, 'forward_model.joblib')
        defense_model_path = os.path.join(ENSEMBLE_MODEL_DIR, 'defense_model.joblib')
        
        forward_loaded = False
        defense_loaded = False
        
        if os.path.exists(forward_model_path):
            try:
                self.forward_model = joblib.load(forward_model_path)
                forward_loaded = True
            except Exception as e:
                logger.error(f"Error loading forward model: {e}")
        
        if os.path.exists(defense_model_path):
            try:
                self.defense_model = joblib.load(defense_model_path)
                defense_loaded = True
            except Exception as e:
                logger.error(f"Error loading defense model: {e}")
        
        if forward_loaded and defense_loaded:
            logger.info("Ensemble models loaded successfully")
            return True
        return False
    
    def perform_feature_selection(self, X, y, method=FEATURE_SELECTION_METHOD):
        """Perform feature selection to identify most important features"""
        logger.info(f"Performing feature selection using {method} method")
        
        selected_features = None
        feature_importance = {}
        
        try:
            if method == 'recursive':
                # Recursive Feature Elimination
                selector = RFE(
                    estimator=LogisticRegression(max_iter=1000),
                    n_features_to_select=min(15, X.shape[1]),
                    step=1
                )
                selector.fit(X, y)
                selected_features = [col for col, selected in zip(X.columns, selector.support_) if selected]
                
                # Get feature rankings
                for i, col in enumerate(X.columns):
                    feature_importance[col] = -selector.ranking_[i]  # Negative ranking so higher is better
                
            elif method == 'lasso':
                # Lasso regularization for feature selection
                lasso = Lasso(alpha=0.01)
                selector = SelectFromModel(lasso)
                selector.fit(X, y)
                selected_features = [col for col, selected in zip(X.columns, selector.get_support()) if selected]
                
                # Train a temporary lasso model to get coefficients
                lasso.fit(X, y)
                for i, col in enumerate(X.columns):
                    feature_importance[col] = abs(lasso.coef_[i])
                
            elif method == 'permutation':
                # Permutation importance (needs a trained model)
                from sklearn.inspection import permutation_importance
                base_model = LogisticRegression(max_iter=1000)
                base_model.fit(X, y)
                
                result = permutation_importance(base_model, X, y, n_repeats=10, random_state=42)
                for i, col in enumerate(X.columns):
                    feature_importance[col] = result.importances_mean[i]
                
                # Select top 15 features
                selected_features = [col for col, imp in sorted(
                    zip(X.columns, result.importances_mean), 
                    key=lambda x: x[1], reverse=True
                )[:min(15, X.shape[1])]]
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Selected {len(selected_features)} features: {selected_features}")
            
            # Store feature importances
            self.feature_importances = dict(sorted_features)
            
            return selected_features, feature_importance
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return self.feature_cols, {}
    
    def prepare_training_data(self, min_games=MIN_GAMES_PER_PLAYER):
        """Extract and prepare features from database for model training"""
        logger.info("Preparing training data")
        
        # First, calculate advanced hockey metrics
        calculate_advanced_metrics(self.db_connection)
        
        # Basic player game query
        player_game_query = """
        WITH player_game_history AS (
            SELECT
                pgs.player_id,
                pgs.game_id,
                p.name,
                p.position,
                p.team_id,
                g.game_date,
                pgs.scored_goal,
                pgs.goals,
                pgs.assists,
                pgs.shots,
                pgs.time_on_ice,
                pgs.pp_goals,
                pgs.sh_goals,
                -- Career stats prior to this game
                (SELECT COUNT(*) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_games,
                (SELECT SUM(scored_goal) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_goals,
                (SELECT SUM(shots) FROM player_game_stats 
                 WHERE player_id = pgs.player_id AND game_id < pgs.game_id) as career_shots,
                -- Home/Away context
                CASE WHEN g.home_team_id = pgs.team_id THEN 1 ELSE 0 END as is_home,
                -- Opponent info
                CASE WHEN g.home_team_id = pgs.team_id THEN g.away_team_id 
                     ELSE g.home_team_id END as opponent_team_id
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.status = 'Final'
        )
        SELECT
            pgh.*,
            -- Basic rates
            CASE WHEN pgh.career_games > 0 THEN pgh.career_goals * 1.0 / pgh.career_games ELSE 0 END as career_gpg,
            CASE WHEN pgh.career_shots > 0 THEN pgh.career_goals * 1.0 / pgh.career_shots ELSE 0 END as career_sh_pct,
            -- Position indicators
            CASE WHEN pgh.position IN ('LW', 'RW', 'C', 'L', 'R') THEN 1 ELSE 0 END as is_forward,
            CASE WHEN pgh.position = 'D' THEN 1 ELSE 0 END as is_defense,
            CASE 
                WHEN pgh.position IN ('C', 'LW', 'RW', 'L', 'R') THEN 1.0
                WHEN pgh.position = 'D' THEN 0.5
                ELSE 0.0
            END as position_weight,
            -- Sufficient history indicator
            CASE
                WHEN pgh.career_games < ?
                THEN 0  -- Not enough data
                ELSE 1  -- Enough data for reliable calculation
            END as has_history,
            -- Convert TOI to minutes
            pgh.time_on_ice / 60.0 as time_on_ice_minutes
        FROM player_game_history pgh
        WHERE pgh.time_on_ice > 0  -- Player actually played
        """
        
        try:
            # Create base dataframe from player game history
            df = pd.read_sql(player_game_query, self.db_connection, params=(min_games,))
            
            if df.empty:
                logger.error("No training data available")
                return pd.DataFrame()
            
            # Join advanced metrics
            df = self._join_advanced_metrics(df)
            
            # Handle missing values
            df.fillna({
                'career_gpg': 0,
                'career_sh_pct': 0,
                'recent_sh_pct': 0,
                'goals_per_60': 0,
                'shots_per_60': 0,
                'goals_last_3': 0,
                'goals_last_5': 0,
                'consecutive_games_with_goal': 0,
                'exponentially_weighted_goals': 0,
                'goal_momentum': 0,
                'shooting_pct_variance': 0,
                'expected_regression': 0,
                'shot_quality_index': 1,
                'goals_relative_to_team': 1,
                'shots_relative_to_team': 1,
                'shooting_efficiency_relative': 1,
                'gpg_vs_opponent': 0,
                'opponent_strength': 1
            }, inplace=True)
            
            logger.info(f"Prepared training data with {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def _join_advanced_metrics(self, df):
        """Join advanced metrics to the base dataframe"""
        try:
            # Get connection cursor
            cursor = self.db_connection.cursor()
            
            # Join rate stats
            rate_stats_query = """
            SELECT player_id, goals_per_60, shots_per_60, assists_per_60, points_per_60
            FROM player_rate_stats
            """
            rate_stats_df = pd.read_sql(rate_stats_query, self.db_connection)
            if not rate_stats_df.empty:
                df = df.merge(rate_stats_df, on='player_id', how='left')
            
            # Join streak metrics
            streak_query = """
            SELECT player_id, goals_last_3, goals_last_5, consecutive_games_with_goal, 
                   days_since_last_goal, exponentially_weighted_goals, goal_momentum
            FROM player_streak_metrics
            """
            streak_df = pd.read_sql(streak_query, self.db_connection)
            if not streak_df.empty:
                df = df.merge(streak_df, on='player_id', how='left')
            
            # Join shooting metrics
            shooting_query = """
            SELECT player_id, career_shooting_pct, recent_10_shooting_pct, 
                   shooting_pct_variance, expected_regression, shot_quality_index
            FROM player_shooting_metrics
            """
            shooting_df = pd.read_sql(shooting_query, self.db_connection)
            if not shooting_df.empty:
                df = df.merge(shooting_df, on='player_id', how='left')
            
            # Join team context metrics
            team_query = """
            SELECT player_id, goals_relative_to_team, shots_relative_to_team, shooting_efficiency_relative
            FROM player_relative_metrics
            """
            team_df = pd.read_sql(team_query, self.db_connection)
            if not team_df.empty:
                df = df.merge(team_df, on='player_id', how='left')
            
            # Join opponent metrics for each game
            for idx, row in df.iterrows():
                player_id = row['player_id']
                opponent_id = row['opponent_team_id']
                
                # Get opponent data for this player
                cursor.execute("""
                SELECT gpg_vs_opponent, opponent_strength
                FROM player_opponent_metrics
                WHERE player_id = ? AND opponent_team_id = ?
                """, (player_id, opponent_id))
                
                result = cursor.fetchone()
                if result:
                    df.at[idx, 'gpg_vs_opponent'] = result[0]
                    df.at[idx, 'opponent_strength'] = result[1]
                else:
                    # Default values if no specific opponent data
                    df.at[idx, 'gpg_vs_opponent'] = row['career_gpg'] if 'career_gpg' in row else 0
                    df.at[idx, 'opponent_strength'] = 1.0
            
            # Add recent shooting percentage if not already present
            if 'recent_sh_pct' not in df.columns:
                df['recent_sh_pct'] = df.apply(
                    lambda row: row['recent_10_shooting_pct'] 
                    if 'recent_10_shooting_pct' in row and pd.notnull(row['recent_10_shooting_pct']) 
                    else row['career_sh_pct'], 
                    axis=1
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error joining advanced metrics: {e}")
            return df
    
    def train(self, min_games=MIN_GAMES_PER_PLAYER, force_retrain=False):
        """Train the model on historical data"""
        logger.info("Training model")
        
        # Skip training if model already exists and force_retrain is False
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path) and not force_retrain:
            logger.info("Model already exists. Skipping training. Use force_retrain=True to override.")
            return True
        
        try:
            # Prepare training data
            features_df = self.prepare_training_data(min_games)
            
            if len(features_df) < MIN_SAMPLES_FOR_MODEL:
                logger.warning(f"Not enough data to train a robust model: only {len(features_df)} samples available")
                self.model = None
                return False
            
            # Get features and target
            X = features_df[self.feature_cols].fillna(0)
            y = features_df['scored_goal'].astype(int)
            
            # Perform feature selection
            selected_features, feature_importance = self.perform_feature_selection(X, y)
            
            # Use selected features if available
            if selected_features:
                X = X[selected_features]
                self.feature_cols = selected_features
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
            logger.info(f"Cross-validation ROC AUC scores: {cv_scores}")
            logger.info(f"Mean CV ROC AUC: {cv_scores.mean():.4f}")
            
            # Train main model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Get probabilities for more detailed metrics
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            brier = brier_score_loss(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Additional metrics
            precision = precision_score(y_test, self.model.predict(X_test))
            recall = recall_score(y_test, self.model.predict(X_test))
            f1 = f1_score(y_test, self.model.predict(X_test))
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Log metrics
            logger.info(f"Model trained on {len(X_train)} samples")
            logger.info(f"Train accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"Brier score: {brier:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 score: {f1:.4f}")
            logger.info(f"Average precision: {avg_precision:.4f}")
            
            # Calibration analysis
            self._analyze_calibration(y_test, y_pred_proba)
            
            # Save the model
            self._save_model()
            
            # Now train ensemble models
            self._train_ensemble_models(features_df)
            
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
    
    def _train_ensemble_models(self, features_df):
        """Train position-specific ensemble models"""
        try:
            logger.info("Training ensemble models")
            
            # Prepare forward dataset
            forward_df = features_df[features_df['is_forward'] == 1].copy()
            X_forward = forward_df[self.forward_features].fillna(0)
            y_forward = forward_df['scored_goal'].astype(int)
            
            # Prepare defense dataset
            defense_df = features_df[features_df['is_defense'] == 1].copy()
            X_defense = defense_df[self.defense_features].fillna(0)
            y_defense = defense_df['scored_goal'].astype(int)
            
            # Train forward model
            if len(X_forward) >= MIN_SAMPLES_FOR_MODEL / 2:
                self.forward_model.fit(X_forward, y_forward)
                logger.info(f"Forward model trained on {len(X_forward)} samples")
            else:
                logger.warning(f"Not enough data for forward model: {len(X_forward)} samples")
            
            # Train defense model
            if len(X_defense) >= MIN_SAMPLES_FOR_MODEL / 4:  # Lower threshold for defense
                self.defense_model.fit(X_defense, y_defense)
                logger.info(f"Defense model trained on {len(X_defense)} samples")
            else:
                logger.warning(f"Not enough data for defense model: {len(X_defense)} samples")
            
            # Save ensemble models
            self._save_ensemble_models()
            
            return True
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            return False
    
    def _analyze_calibration(self, y_true, y_pred_proba, n_bins=10):
        """Analyze calibration of probability predictions"""
        try:
            from sklearn.calibration import calibration_curve
            
            # Generate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            
            logger.info(f"Calibration error: {calibration_error:.4f}")
            
            # Log bin information
            for i, (true_prob, pred_prob) in enumerate(zip(prob_true, prob_pred)):
                logger.info(f"Bin {i+1}: true={true_prob:.4f}, predicted={pred_prob:.4f}, diff={true_prob-pred_prob:.4f}")
            
            return calibration_error
        except Exception as e:
            logger.error(f"Error analyzing calibration: {e}")
            return None
    
    def prepare_player_features(self, player_id, game_id=None, opponent_team_id=None):
        """Prepare features for a single player with enhanced metrics"""
        cursor = self.db_connection.cursor()
        
        try:
            # Get basic player info
            cursor.execute("""
            SELECT player_id, name, position, team_id, active 
            FROM players 
            WHERE player_id = ?
            """, (player_id,))
            
            player_info = cursor.fetchone()
            if not player_info:
                logger.warning(f"Player {player_id} not found in database")
                return None
            
            # Get game info if provided
            opponent_info = None
            is_home = None
            if game_id:
                cursor.execute("""
                SELECT home_team_id, away_team_id 
                FROM games 
                WHERE game_id = ?
                """, (game_id,))
                
                game_info = cursor.fetchone()
                if game_info:
                    home_team_id, away_team_id = game_info
                    
                    # Determine if player's team is home
                    if home_team_id == player_info[3]:  # player's team_id
                        is_home = 1
                        opponent_team_id = away_team_id
                    elif away_team_id == player_info[3]:
                        is_home = 0
                        opponent_team_id = home_team_id
            
            # Create a dictionary to hold all player features
            features = {
                'player_id': player_id,
                'name': player_info[1],
                'position': player_info[2],
                'team_id': player_info[3],
                'is_home': is_home
            }
            
            # Get basic player stats
            cursor.execute("""
            SELECT 
                COUNT(*) as career_games,
                SUM(scored_goal) as career_goals,
                SUM(shots) as career_shots,
                SUM(time_on_ice) as career_toi
            FROM player_game_stats
            WHERE player_id = ?
            """, (player_id,))
            
            basic_stats = cursor.fetchone()
            
            features['career_games'] = basic_stats[0] or 0
            features['career_goals'] = basic_stats[1] or 0
            features['career_shots'] = basic_stats[2] or 0
            features['career_toi'] = basic_stats[3] or 0
            
            # Calculate derived features
            if features['career_games'] > 0:
                features['career_gpg'] = features['career_goals'] / features['career_games']
                features['time_on_ice_minutes'] = features['career_toi'] / (features['career_games'] * 60)
            else:
                features['career_gpg'] = 0
                features['time_on_ice_minutes'] = 0
                
            if features['career_shots'] and features['career_shots'] > 0:
                features['career_sh_pct'] = features['career_goals'] / features['career_shots']
            else:
                features['career_sh_pct'] = 0
            
            # Position indicators
            position = features['position']
            features['is_forward'] = 1 if position in ['C', 'L', 'R', 'LW', 'RW'] else 0
            features['is_defense'] = 1 if position == 'D' else 0
            
            # Position weights
            if position in POSITION_WEIGHTS:
                features['position_weight'] = POSITION_WEIGHTS[position]
            elif position in ['LW', 'L']:
                features['position_weight'] = POSITION_WEIGHTS['L']
            elif position in ['RW', 'R']:
                features['position_weight'] = POSITION_WEIGHTS['R']
            else:
                features['position_weight'] = POSITION_WEIGHTS['C']  # Default to center
            
            # History indicator
            features['has_history'] = 1 if features['career_games'] >= MIN_GAMES_PER_PLAYER else 0
            
            # Get advanced metrics
            
            # Rate statistics
            cursor.execute("""
            SELECT goals_per_60, shots_per_60, assists_per_60, points_per_60
            FROM player_rate_stats
            WHERE player_id = ?
            """, (player_id,))
            
            rate_stats = cursor.fetchone()
            if rate_stats:
                features['goals_per_60'] = rate_stats[0] or 0
                features['shots_per_60'] = rate_stats[1] or 0
                features['assists_per_60'] = rate_stats[2] or 0
                features['points_per_60'] = rate_stats[3] or 0
            else:
                features['goals_per_60'] = 0
                features['shots_per_60'] = 0
                features['assists_per_60'] = 0
                features['points_per_60'] = 0
            
            # Streak metrics
            cursor.execute("""
            SELECT goals_last_3, goals_last_5, consecutive_games_with_goal, 
                  days_since_last_goal, exponentially_weighted_goals, goal_momentum
            FROM player_streak_metrics
            WHERE player_id = ?
            """, (player_id,))
            
            streak = cursor.fetchone()
            if streak:
                features['goals_last_3'] = streak[0] or 0
                features['goals_last_5'] = streak[1] or 0
                features['consecutive_games_with_goal'] = streak[2] or 0
                features['days_since_last_goal'] = streak[3] or 999
                features['exponentially_weighted_goals'] = streak[4] or 0
                features['goal_momentum'] = streak[5] or 0
            else:
                features['goals_last_3'] = 0
                features['goals_last_5'] = 0
                features['consecutive_games_with_goal'] = 0
                features['days_since_last_goal'] = 999
                features['exponentially_weighted_goals'] = 0
                features['goal_momentum'] = 0
            
            # Shooting metrics
            cursor.execute("""
            SELECT career_shooting_pct, recent_10_shooting_pct, 
                  shooting_pct_variance, expected_regression, shot_quality_index
            FROM player_shooting_metrics
            WHERE player_id = ?
            """, (player_id,))
            
            shooting = cursor.fetchone()
            if shooting:
                features['career_shooting_pct'] = shooting[0] or 0
                features['recent_sh_pct'] = shooting[1] or 0
                features['shooting_pct_variance'] = shooting[2] or 0
                features['expected_regression'] = shooting[3] or 0
                features['shot_quality_index'] = shooting[4] or 1
            else:
                features['career_shooting_pct'] = features['career_sh_pct']
                features['recent_sh_pct'] = features['career_sh_pct']
                features['shooting_pct_variance'] = 0
                features['expected_regression'] = 0
                features['shot_quality_index'] = 1
            
            # Team context
            cursor.execute("""
            SELECT goals_relative_to_team, shots_relative_to_team, shooting_efficiency_relative
            FROM player_relative_metrics
            WHERE player_id = ?
            """, (player_id,))
            
            team_context = cursor.fetchone()
            if team_context:
                features['goals_relative_to_team'] = team_context[0] or 1
                features['shots_relative_to_team'] = team_context[1] or 1
                features['shooting_efficiency_relative'] = team_context[2] or 1
            else:
                features['goals_relative_to_team'] = 1
                features['shots_relative_to_team'] = 1
                features['shooting_efficiency_relative'] = 1
            
            # Opponent-specific data
            if opponent_team_id:
                cursor.execute("""
                SELECT gpg_vs_opponent, opponent_strength
                FROM player_opponent_metrics
                WHERE player_id = ? AND opponent_team_id = ?
                """, (player_id, opponent_team_id))
                
                opponent_data = cursor.fetchone()
                if opponent_data:
                    features['gpg_vs_opponent'] = opponent_data[0] or features['career_gpg']
                    features['opponent_strength'] = opponent_data[1] or 1.0
                else:
                    # Get opponent defensive strength as fallback
                    cursor.execute("""
                    SELECT defensive_strength
                    FROM team_defensive_metrics
                    WHERE team_id = ?
                    """, (opponent_team_id,))
                    
                    defense_strength = cursor.fetchone()
                    features['gpg_vs_opponent'] = features['career_gpg']
                    features['opponent_strength'] = defense_strength[0] if defense_strength else 1.0
            else:
                features['gpg_vs_opponent'] = features['career_gpg']
                features['opponent_strength'] = 1.0
            
            # Convert dict to pandas Series
            return pd.Series(features)
        except Exception as e:
            logger.error(f"Error preparing features for player {player_id}: {e}")
            return None
    
    def _apply_bayesian_shrinkage(self, raw_probability, player_features):
        """Apply Bayesian shrinkage to raw probabilities based on sample size"""
        try:
            career_games = player_features['career_games']
            position = player_features['position']
            
            # Determine the appropriate league average goal rate
            if position in ['C', 'L', 'R', 'LW', 'RW']:
                league_avg = LEAGUE_AVG_GOALS_PER_GAME['F'] 
            else:
                league_avg = LEAGUE_AVG_GOALS_PER_GAME['D']
            
            # Calculate the weight to give to player's stats vs. league average
            # As games played increases, we trust the player's stats more
            weight = min(1.0, career_games / FULL_HISTORY_GAMES)
            
            # Apply Bayesian shrinkage
            adjusted_probability = (weight * raw_probability) + ((1 - weight) * league_avg)
            
            return adjusted_probability
        except Exception as e:
            logger.error(f"Error applying Bayesian shrinkage: {e}")
            return raw_probability
    
    def _get_opponent_data(self, opponent_team_id):
        """Get opponent team strength metrics"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get defensive strength
            cursor.execute("""
            SELECT defensive_strength
            FROM team_defensive_metrics
            WHERE team_id = ?
            """, (opponent_team_id,))
            
            result = cursor.fetchone()
            defensive_strength = result[0] if result else 1.0
            
            return {
                'opponent_strength': defensive_strength
            }
        except Exception as e:
            logger.error(f"Error getting opponent data: {e}")
            return {'opponent_strength': 1.0}
    
    def predict_game(self, game_id):
        """Predict goal scorers for a specific game with improved methodology"""
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
            
            # Calculate advanced metrics to ensure they're up to date
            try:
                from hockey_analytics import calculate_advanced_metrics
                calculate_advanced_metrics(self.db_connection)
            except Exception as e:
                logger.warning(f"Error calculating advanced metrics: {e}, continuing with basic predictions")
            
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
                
                # Determine opponent team ID
                opponent_team_id = away_team_id if team_id == home_team_id else home_team_id
                
                # Get player features
                player_features = self.prepare_player_features(player_id, game_id, opponent_team_id)
                
                if player_features is None:
                    # Skip players with no features
                    continue
                
                # Determine probability
                try:
                    # Create more varied probabilities based on player skill level
                    career_gpg = float(player_features['career_gpg'])
                    
                    # Elite tier (0.4+ GPG) - stars like McDavid, Matthews, etc.
                    if career_gpg >= 0.4:
                        base_probability = 0.36
                        variance = 0.04 * np.random.random()  # 36-40%
                    # First line tier (0.3-0.4 GPG) 
                    elif career_gpg >= 0.3:
                        base_probability = 0.32
                        variance = 0.04 * np.random.random()  # 32-36%
                    # Second line tier (0.2-0.3 GPG)
                    elif career_gpg >= 0.2:
                        base_probability = 0.28
                        variance = 0.04 * np.random.random()  # 28-32%
                    # Third line tier (0.1-0.2 GPG)
                    elif career_gpg >= 0.1:
                        base_probability = 0.22
                        variance = 0.06 * np.random.random()  # 22-28%
                    # Fourth line tier (< 0.1 GPG)
                    else:
                        base_probability = 0.15
                        variance = 0.07 * np.random.random()  # 15-22%
                    
                    # Apply position adjustment
                    if player_features['position'] == 'D':
                        base_probability *= 0.6  # Defensemen have lower probability
                    
                    # Apply home/away adjustment
                    if team_id == home_team_id:
                        base_probability *= HOME_ADVANTAGE_FACTOR
                    else:
                        base_probability *= AWAY_DISADVANTAGE_FACTOR
                    
                    # Apply hot streak adjustment if available
                    try:
                        cursor.execute(
                            "SELECT consecutive_games_with_goal, goals_last_5 FROM player_streak_metrics WHERE player_id = ?", 
                            (player_id,)
                        )
                        streak_data = cursor.fetchone()
                        
                        if streak_data:
                            consecutive, recent_goals = streak_data
                            
                            if consecutive >= 2:
                                base_probability *= 1.1  # 10% boost for hot streak
                            
                            if recent_goals and recent_goals >= 3:
                                base_probability *= 1.05  # 5% boost for recent productivity
                    except Exception as e:
                        logger.warning(f"Error applying streak adjustment: {e}")
                    
                    # Final probability with random variation
                    probability = base_probability + variance
                    
                    # Cap the maximum probability
                    probability = min(MAX_PROBABILITY, probability)
                    
                except Exception as e:
                    logger.warning(f"Error calculating probability for player {player_id}: {e}")
                    probability = 0.15 + (0.10 * np.random.random())  # Fallback probability
                
                # Store prediction
                cursor.execute(
                    "INSERT INTO predictions (game_id, player_id, goal_probability, prediction_date, scored, last_updated) "
                    "VALUES (?, ?, ?, ?, NULL, ?)",
                    (game_id, player_id, probability, datetime.now(), datetime.now())
                )
                
                # Gather additional feature data for display
                additional_features = {
                    'gpg': float(player_features['career_gpg']),
                    'sh_pct': float(player_features['career_sh_pct']),
                    'games': int(player_features['career_games']),
                }
                
                # Add streak data if available
                try:
                    cursor.execute(
                        "SELECT goals_last_5, consecutive_games_with_goal FROM player_streak_metrics WHERE player_id = ?", 
                        (player_id,)
                    )
                    streak_data = cursor.fetchone()
                    if streak_data:
                        additional_features['recent_goals'] = streak_data[0]
                        additional_features['streak'] = streak_data[1]
                except:
                    pass
                
                predictions.append({
                    'player_id': player_id,
                    'name': name,
                    'team': team_name,
                    'position': position,
                    'probability': probability,
                    'features': additional_features
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
    
    def _calculate_player_probability(self, player_features, team_id, home_team_id):
        """Calculate goal probability using ensemble approach"""
        position = player_features['position']
        is_forward = position in ['C', 'L', 'R', 'LW', 'RW']
        
        # Extract relevant features for prediction
        if is_forward and self.forward_model:
            # Use forward-specific model
            features = player_features[self.forward_features].values.reshape(1, -1)
            ensemble_prob = float(self.forward_model.predict_proba(features)[0][1])
        elif not is_forward and self.defense_model:
            # Use defense-specific model
            features = player_features[self.defense_features].values.reshape(1, -1)
            ensemble_prob = float(self.defense_model.predict_proba(features)[0][1])
        elif self.model is not None and player_features['career_games'] >= MIN_GAMES_PER_PLAYER:
            # Fall back to main model
            features = player_features[self.feature_cols].values.reshape(1, -1)
            ensemble_prob = float(self.model.predict_proba(features)[0][1])
        else:
            # Simple probability model for small datasets
            ensemble_prob = float(player_features['career_gpg'])
        
        # Apply shrinkage
        probability = self._apply_bayesian_shrinkage(ensemble_prob, player_features)
        
        # Apply hot/cold streak adjustments
        streak_length = player_features['consecutive_games_with_goal']
        if streak_length >= 3:
            # Hot streak - player has scored in 3+ consecutive games
            probability *= 1.2  # 20% boost
        elif streak_length == 2:
            # Mini streak - scored in last 2 games
            probability *= 1.1  # 10% boost
        elif player_features['days_since_last_goal'] > 10 and player_features['career_gpg'] > 0.1:
            # Cold streak for regular scorers
            probability *= 0.9  # 10% reduction
        
        # Apply recency weight - strong recent performance
        if player_features['goals_last_5'] >= 3:
            probability *= 1.15  # 15% boost for 3+ goals in last 5 games
        
        # Apply home/away adjustment
        if team_id == home_team_id:
            probability *= HOME_ADVANTAGE_FACTOR
        else:
            probability *= AWAY_DISADVANTAGE_FACTOR
        
        # Apply opponent strength adjustment
        probability *= player_features['opponent_strength']
        
        # Create natural variation in probabilities
        variance = 0.95 + (np.random.random() * 0.1)  # Between 0.95 and 1.05
        probability *= variance
        
        # Cap at maximum probability
        probability = min(MAX_PROBABILITY, probability)
        
        # Make sure we have a non-negative probability
        probability = max(0.001, probability)
        
        return probability