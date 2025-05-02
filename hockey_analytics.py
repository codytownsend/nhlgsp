"""
hockey_analytics.py - Advanced hockey statistics and metrics
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sqlite3

logger = logging.getLogger(__name__)

def calculate_advanced_metrics(conn):
    """Calculate and store advanced hockey metrics for all players"""
    logger.info("Calculating advanced hockey metrics")
    
    try:
        # First, ensure tables have the correct schema
        ensure_correct_tables_schema(conn)
        
        # Calculate rate statistics
        calculate_rate_stats(conn)
        
        # Calculate streak metrics
        calculate_streak_metrics(conn)
        
        # Calculate shooting metrics
        calculate_shooting_metrics(conn)
        
        logger.info("Advanced metrics calculation complete")
    except Exception as e:
        logger.error(f"Error in calculate_advanced_metrics: {e}")

def ensure_correct_tables_schema(conn):
    """Ensure all tables have the correct schema"""
    cursor = conn.cursor()
    
    try:
        # Drop and recreate player_shooting_metrics with the correct schema
        cursor.execute("DROP TABLE IF EXISTS player_shooting_metrics")
        
        cursor.execute('''
        CREATE TABLE player_shooting_metrics (
            player_id INTEGER PRIMARY KEY,
            career_shooting_pct REAL,
            recent_10_shooting_pct REAL,
            shooting_pct_variance REAL,
            expected_regression REAL,
            shot_quality_index REAL,
            last_updated TIMESTAMP
        )
        ''')
        
        # Ensure player_rate_stats has the correct schema
        cursor.execute("DROP TABLE IF EXISTS player_rate_stats")
        
        cursor.execute('''
        CREATE TABLE player_rate_stats (
            player_id INTEGER PRIMARY KEY,
            goals_per_60 REAL,
            shots_per_60 REAL,
            assists_per_60 REAL,
            points_per_60 REAL,
            last_updated TIMESTAMP
        )
        ''')
        
        # Ensure player_streak_metrics has the correct schema
        cursor.execute("DROP TABLE IF EXISTS player_streak_metrics")
        
        cursor.execute('''
        CREATE TABLE player_streak_metrics (
            player_id INTEGER PRIMARY KEY,
            goals_last_3 INTEGER,
            goals_last_5 INTEGER,
            goals_last_10 INTEGER,
            consecutive_games_with_goal INTEGER,
            days_since_last_goal INTEGER,
            exponentially_weighted_goals REAL,
            goal_momentum REAL,
            last_updated TIMESTAMP
        )
        ''')
        
        conn.commit()
        logger.info("Table schemas updated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating table schemas: {e}")
        raise

def calculate_rate_stats(conn):
    """Calculate rate statistics per 60 minutes of ice time"""
    cursor = conn.cursor()
    
    try:
        # Calculate rate stats
        cursor.execute('''
        INSERT OR REPLACE INTO player_rate_stats 
        (player_id, goals_per_60, shots_per_60, assists_per_60, points_per_60, last_updated)
        SELECT 
            player_id,
            SUM(goals) * 60.0 * 60.0 / NULLIF(SUM(time_on_ice), 0) as goals_per_60,
            SUM(shots) * 60.0 * 60.0 / NULLIF(SUM(time_on_ice), 0) as shots_per_60,
            SUM(assists) * 60.0 * 60.0 / NULLIF(SUM(time_on_ice), 0) as assists_per_60,
            (SUM(goals) + SUM(assists)) * 60.0 * 60.0 / NULLIF(SUM(time_on_ice), 0) as points_per_60,
            datetime('now')
        FROM player_game_stats
        GROUP BY player_id
        ''')
        
        conn.commit()
        logger.info("Rate statistics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating rate stats: {e}")

def calculate_streak_metrics(conn):
    """Calculate hot/cold streaks and momentum metrics"""
    cursor = conn.cursor()
    
    try:
        # Get all players
        cursor.execute("SELECT DISTINCT player_id FROM players WHERE active = 1")
        players = cursor.fetchall()
        
        for player_id, in players:
            try:
                # Get player's recent games in chronological order
                cursor.execute('''
                SELECT pgs.game_id, g.game_date, pgs.scored_goal
                FROM player_game_stats pgs
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.player_id = ? AND g.game_date IS NOT NULL
                ORDER BY g.game_date DESC
                LIMIT 20
                ''', (player_id,))
                
                games = cursor.fetchall()
                
                if not games:
                    continue
                    
                # Calculate metrics
                goals_last_3 = sum(game[2] for game in games[:3] if game[2] is not None)
                goals_last_5 = sum(game[2] for game in games[:5] if game[2] is not None)
                goals_last_10 = sum(game[2] for game in games[:10] if game[2] is not None)
                
                # Calculate consecutive games with a goal
                consecutive = 0
                for game in games:
                    if game[2] == 1:
                        consecutive += 1
                    else:
                        break
                
                # Calculate days since last goal
                days_since_last_goal = 999  # Default if no goal found
                for game in games:
                    if game[2] == 1 and game[1] is not None:
                        try:
                            last_goal_date = datetime.strptime(game[1], '%Y-%m-%d')
                            days_since_last_goal = (datetime.now() - last_goal_date).days
                        except (ValueError, TypeError):
                            days_since_last_goal = 999
                        break
                
                # Calculate exponentially weighted goals (more weight to recent games)
                exp_weighted_goals = 0
                weights_sum = 0
                for i, game in enumerate(games[:10]):
                    if game[2] is not None:
                        weight = np.exp(-0.3 * i)  # Exponential decay
                        exp_weighted_goals += game[2] * weight
                        weights_sum += weight
                
                if weights_sum > 0:
                    exp_weighted_goals /= weights_sum
                
                # Calculate goal momentum (trend in last 10 games vs previous 10)
                recent_10 = sum(game[2] for game in games[:10] if game[2] is not None)
                previous_10 = sum(game[2] for game in games[10:20] if game[2] is not None)
                
                if len(games) >= 20:
                    goal_momentum = recent_10 - previous_10
                else:
                    goal_momentum = 0
                
                # Store calculated metrics
                cursor.execute('''
                INSERT OR REPLACE INTO player_streak_metrics
                (player_id, goals_last_3, goals_last_5, goals_last_10, 
                 consecutive_games_with_goal, days_since_last_goal, 
                 exponentially_weighted_goals, goal_momentum, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (player_id, goals_last_3, goals_last_5, goals_last_10, 
                      consecutive, days_since_last_goal, 
                      exp_weighted_goals, goal_momentum))
            
            except Exception as e:
                logger.error(f"Error processing streak metrics for player {player_id}: {e}")
                # Continue with next player
                continue
        
        conn.commit()
        logger.info("Streak metrics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating streak metrics: {e}")

def calculate_shooting_metrics(conn):
    """Calculate advanced shooting metrics"""
    cursor = conn.cursor()
    
    try:
        # Get all players
        cursor.execute("SELECT DISTINCT player_id FROM players WHERE active = 1")
        players = cursor.fetchall()
        
        for player_id, in players:
            try:
                # Get career shooting data
                cursor.execute('''
                SELECT SUM(goals), SUM(shots)
                FROM player_game_stats
                WHERE player_id = ?
                ''', (player_id,))
                
                career_goals, career_shots = cursor.fetchone()
                
                if not career_goals or not career_shots or career_shots == 0:
                    continue
                
                career_shooting_pct = career_goals / career_shots
                
                # Get recent shooting data
                cursor.execute('''
                SELECT pgs.goals, pgs.shots
                FROM player_game_stats pgs
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.player_id = ? AND g.game_date IS NOT NULL
                ORDER BY g.game_date DESC
                LIMIT 10
                ''', (player_id,))
                
                recent_games = cursor.fetchall()
                
                recent_goals = sum(game[0] for game in recent_games if game[0] is not None)
                recent_shots = sum(game[1] for game in recent_games if game[1] is not None)
                
                if recent_shots > 0:
                    recent_shooting_pct = recent_goals / recent_shots
                else:
                    recent_shooting_pct = 0
                
                # Calculate shooting percentage variance
                cursor.execute('''
                SELECT g.season, SUM(pgs.goals), SUM(pgs.shots)
                FROM player_game_stats pgs
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.player_id = ? AND g.season IS NOT NULL
                GROUP BY g.season
                HAVING SUM(pgs.shots) > 10
                ''', (player_id,))
                
                season_data = cursor.fetchall()
                
                if len(season_data) > 1:
                    season_sh_pcts = [goals/shots for _, goals, shots in season_data if shots > 0]
                    shooting_pct_variance = np.var(season_sh_pcts) if len(season_sh_pcts) > 1 else 0
                else:
                    shooting_pct_variance = 0
                
                # Calculate expected regression (towards career mean)
                expected_regression = (career_shooting_pct - recent_shooting_pct) if recent_shooting_pct > 0 else 0
                
                # Calculate shot quality index (proxy for expected goals)
                # Higher value = better shot quality
                shot_quality_index = career_shooting_pct / 0.095  # League average shooting % is ~9.5%
                
                # Store calculated metrics
                cursor.execute('''
                INSERT OR REPLACE INTO player_shooting_metrics
                (player_id, career_shooting_pct, recent_10_shooting_pct, 
                 shooting_pct_variance, expected_regression, shot_quality_index, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (player_id, career_shooting_pct, recent_shooting_pct, 
                      shooting_pct_variance, expected_regression, shot_quality_index))
            
            except Exception as e:
                logger.error(f"Error processing shooting metrics for player {player_id}: {e}")
                # Continue with next player
                continue
        
        conn.commit()
        logger.info("Shooting metrics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating shooting metrics: {e}")