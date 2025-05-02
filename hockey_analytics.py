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
        
        # Calculate team-relative metrics
        calculate_relative_metrics(conn)
        
        # Calculate team defensive metrics
        calculate_team_defensive_metrics(conn)
        
        # Calculate opponent-specific metrics
        calculate_opponent_metrics(conn)
        
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
        
        # Create player_relative_metrics table
        cursor.execute("DROP TABLE IF EXISTS player_relative_metrics")
        
        cursor.execute('''
        CREATE TABLE player_relative_metrics (
            player_id INTEGER PRIMARY KEY,
            goals_relative_to_team REAL,
            shots_relative_to_team REAL,
            shooting_efficiency_relative REAL,
            last_updated TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        )
        ''')
        
        # Create player_opponent_metrics table
        cursor.execute("DROP TABLE IF EXISTS player_opponent_metrics")
        
        cursor.execute('''
        CREATE TABLE player_opponent_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            opponent_team_id INTEGER,
            gpg_vs_opponent REAL,
            opponent_strength REAL,
            last_updated TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (player_id),
            FOREIGN KEY (opponent_team_id) REFERENCES teams (team_id),
            UNIQUE(player_id, opponent_team_id)
        )
        ''')
        
        # Create team_defensive_metrics table
        cursor.execute("DROP TABLE IF EXISTS team_defensive_metrics")
        
        cursor.execute('''
        CREATE TABLE team_defensive_metrics (
            team_id INTEGER PRIMARY KEY,
            defensive_strength REAL,
            goals_against_per_game REAL,
            shots_against_per_game REAL,
            last_updated TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams (team_id)
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

def calculate_relative_metrics(conn):
    """Calculate team-relative metrics for players"""
    cursor = conn.cursor()
    
    try:
        # Get all players
        cursor.execute("SELECT DISTINCT player_id, team_id FROM players WHERE active = 1")
        players = cursor.fetchall()
        
        # Process each player
        for player_id, team_id in players:
            try:
                if not team_id:
                    continue
                    
                # Get player's goal and shot stats
                cursor.execute('''
                SELECT SUM(goals), SUM(shots), COUNT(*)
                FROM player_game_stats
                WHERE player_id = ?
                ''', (player_id,))
                
                player_stats = cursor.fetchone()
                if not player_stats[0]:
                    continue
                    
                player_goals, player_shots, player_games = player_stats
                
                # Get team average stats (excluding this player)
                cursor.execute('''
                SELECT AVG(goals), AVG(shots)
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE p.team_id = ? AND p.player_id != ?
                GROUP BY p.position
                HAVING p.position = (SELECT position FROM players WHERE player_id = ?)
                ''', (team_id, player_id, player_id))
                
                team_stats = cursor.fetchone()
                if not team_stats:
                    # Fallback to league averages if team stats not available
                    if player_goals > 0 and player_shots > 0:
                        goals_relative = 1.0
                        shots_relative = 1.0
                        efficiency_relative = 1.0
                    else:
                        goals_relative = 0.9
                        shots_relative = 0.9
                        efficiency_relative = 0.9
                else:
                    team_avg_goals, team_avg_shots = team_stats
                    
                    # Calculate relative metrics
                    if team_avg_goals > 0 and player_games > 0:
                        goals_relative = (player_goals / player_games) / team_avg_goals
                    else:
                        goals_relative = 1.0
                        
                    if team_avg_shots > 0 and player_games > 0:
                        shots_relative = (player_shots / player_games) / team_avg_shots
                    else:
                        shots_relative = 1.0
                    
                    # Shooting efficiency relative to teammates
                    if team_avg_shots > 0 and team_avg_goals > 0 and player_shots > 0:
                        player_efficiency = player_goals / player_shots
                        team_efficiency = team_avg_goals / team_avg_shots
                        efficiency_relative = player_efficiency / team_efficiency if team_efficiency > 0 else 1.0
                    else:
                        efficiency_relative = 1.0
                
                # Insert into database
                cursor.execute('''
                INSERT OR REPLACE INTO player_relative_metrics
                (player_id, goals_relative_to_team, shots_relative_to_team, 
                 shooting_efficiency_relative, last_updated)
                VALUES (?, ?, ?, ?, datetime('now'))
                ''', (player_id, goals_relative, shots_relative, efficiency_relative))
                
            except Exception as e:
                logger.error(f"Error calculating relative metrics for player {player_id}: {e}")
                # Continue with next player
                continue
        
        conn.commit()
        logger.info("Relative metrics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating relative metrics: {e}")

def calculate_team_defensive_metrics(conn):
    """Calculate defensive metrics for each team"""
    cursor = conn.cursor()
    
    try:
        # Get all teams
        cursor.execute("SELECT team_id, name FROM teams")
        teams = cursor.fetchall()
        
        for team_id, team_name in teams:
            try:
                # Calculate goals against per game
                cursor.execute('''
                SELECT AVG(
                    CASE 
                        WHEN g.home_team_id = ? THEN g.away_score
                        WHEN g.away_team_id = ? THEN g.home_score
                        ELSE NULL
                    END
                )
                FROM games g
                WHERE (g.home_team_id = ? OR g.away_team_id = ?) AND g.status = 'Final'
                ''', (team_id, team_id, team_id, team_id))
                
                goals_against = cursor.fetchone()[0]
                goals_against = goals_against if goals_against is not None else 3.0  # Default
                
                # Calculate shots against (approximate, as we may not have this directly)
                # Using league average shot-to-goal ratio of about 10%
                shots_against = goals_against * 10.0
                
                # Calculate defensive strength metric (lower goals against = higher strength)
                # Center around 1.0 (higher = better defense)
                defensive_strength = 3.0 / goals_against if goals_against > 0 else 1.0
                
                # Insert into database
                cursor.execute('''
                INSERT OR REPLACE INTO team_defensive_metrics
                (team_id, defensive_strength, goals_against_per_game, 
                 shots_against_per_game, last_updated)
                VALUES (?, ?, ?, ?, datetime('now'))
                ''', (team_id, defensive_strength, goals_against, shots_against))
                
            except Exception as e:
                logger.error(f"Error calculating defensive metrics for team {team_id}: {e}")
                # Continue with next team
                continue
        
        conn.commit()
        logger.info("Team defensive metrics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating team defensive metrics: {e}")

def calculate_opponent_metrics(conn):
    """Calculate player performance metrics against specific opponents"""
    cursor = conn.cursor()
    
    try:
        # Get all active players
        cursor.execute("SELECT player_id FROM players WHERE active = 1")
        players = cursor.fetchall()
        
        # Get all teams
        cursor.execute("SELECT team_id FROM teams")
        teams = cursor.fetchall()
        
        # For each player, calculate metrics against each opponent
        for player_id, in players:
            player_metrics = {}
            
            # Get overall career goals per game for baseline
            cursor.execute('''
            SELECT COUNT(*) AS games, SUM(scored_goal) AS goals
            FROM player_game_stats
            WHERE player_id = ?
            ''', (player_id,))
            
            result = cursor.fetchone()
            total_games, total_goals = result
            
            career_gpg = total_goals / total_games if total_games and total_games > 0 else 0.1
            
            # Process each opponent team
            for team_id, in teams:
                try:
                    # Get defensive strength of opponent
                    cursor.execute('''
                    SELECT defensive_strength 
                    FROM team_defensive_metrics
                    WHERE team_id = ?
                    ''', (team_id,))
                    
                    def_result = cursor.fetchone()
                    defensive_strength = def_result[0] if def_result else 1.0
                    
                    # Get player's performance against this opponent
                    cursor.execute('''
                    SELECT COUNT(*) AS games, SUM(pgs.scored_goal) AS goals
                    FROM player_game_stats pgs
                    JOIN games g ON pgs.game_id = g.game_id
                    WHERE pgs.player_id = ? AND 
                          ((g.home_team_id = ? AND pgs.team_id != ?) OR
                           (g.away_team_id = ? AND pgs.team_id != ?))
                    ''', (player_id, team_id, team_id, team_id, team_id))
                    
                    opp_result = cursor.fetchone()
                    
                    if opp_result and opp_result[0] and opp_result[0] > 0:
                        opp_games, opp_goals = opp_result
                        gpg_vs_opponent = opp_goals / opp_games if opp_goals is not None else career_gpg
                    else:
                        # If no games against this opponent, use career average
                        gpg_vs_opponent = career_gpg
                    
                    # Adjust for opponent strength (stronger opponents = more impressive performance)
                    opponent_strength = 1.0 / defensive_strength if defensive_strength > 0 else 1.0
                    
                    # Insert into database
                    cursor.execute('''
                    INSERT OR REPLACE INTO player_opponent_metrics
                    (player_id, opponent_team_id, gpg_vs_opponent, opponent_strength, last_updated)
                    VALUES (?, ?, ?, ?, datetime('now'))
                    ''', (player_id, team_id, gpg_vs_opponent, opponent_strength))
                
                except Exception as e:
                    logger.error(f"Error processing opponent metrics for player {player_id} vs team {team_id}: {e}")
                    # Insert default values instead of continuing
                    cursor.execute('''
                    INSERT OR REPLACE INTO player_opponent_metrics
                    (player_id, opponent_team_id, gpg_vs_opponent, opponent_strength, last_updated)
                    VALUES (?, ?, ?, ?, datetime('now'))
                    ''', (player_id, team_id, career_gpg, 1.0))
        
        conn.commit()
        logger.info("Opponent metrics calculated successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error calculating opponent metrics: {e}")