import requests
import json
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nhl_data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NHLDataCollector:
    def __init__(self, db_connection):
        # The NHL API base URLs
        self.web_base_url = "https://api-web.nhle.com/v1"
        self.stats_base_url = "https://api.nhle.com/stats/rest/en"
        self.db_connection = db_connection

    def _make_api_request(self, url, params=None, max_retries=3, retry_delay=5):
        """Make an API request with retry logic"""
        retries = 0
        
        # Add headers to better simulate a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'
        }
        
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed: {e}")
                retries += 1
                if retries < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to make API request after {max_retries} attempts: {url}")
                    # Return empty dict to allow fallback behavior rather than raising exception
                    return {}
    
    def get_teams(self):
        """Get all NHL teams"""
        try:
            # Try using the standings endpoint to get team information
            url = f"{self.web_base_url}/standings/now"
            standings_data = self._make_api_request(url)
            
            # Extract team information from standings
            teams_data = {"teams": []}
            if 'standings' in standings_data:
                for standing in standings_data.get('standings', []):
                    team = {
                        'id': standing.get('teamAbbrev', {}).get('id'),
                        'name': standing.get('teamName', {}).get('default'),
                        'abbrev': standing.get('teamAbbrev', {}).get('default')
                    }
                    teams_data["teams"].append(team)
            
            # If we didn't get teams from standings, try the team endpoint
            if not teams_data["teams"]:
                url = f"{self.stats_base_url}/team"
                team_data = self._make_api_request(url)
                if 'data' in team_data:
                    for team in team_data['data']:
                        teams_data["teams"].append({
                            'id': team.get('id'),
                            'name': team.get('fullName'),
                            'abbrev': team.get('triCode')
                        })
            
            return teams_data
        except Exception as e:
            logger.error(f"Error getting teams: {e}")
            # Fallback to empty teams list
            return {"teams": []}
        
    def get_schedule(self, start_date, end_date):
        """Get NHL schedule between two dates"""
        try:
            # For historical data, attempt to fetch each date individually
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            
            combined_data = {"games": []}
            current_date = start_date_obj
            
            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Try the date-specific endpoint
                url = f"{self.web_base_url}/schedule/{date_str}"
                daily_data = self._make_api_request(url)
                
                # Extract games from the nested gameWeek structure
                if "gameWeek" in daily_data:
                    for day in daily_data["gameWeek"]:
                        # Check if this day entry has games
                        if "games" in day and day["games"]:
                            # Add all games from this day
                            combined_data["games"].extend(day["games"])
                
                # Move to next day
                current_date += timedelta(days=1)
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
            
            return combined_data
        except Exception as e:
            logger.error(f"Error getting schedule: {e}")
            return {"games": []}
    
    def get_game_data(self, game_id):
        """Get detailed data for a specific game"""
        url = f"{self.web_base_url}/gamecenter/{game_id}/landing"
        return self._make_api_request(url)
    
    def get_game_boxscore(self, game_id):
        """Get boxscore data for a specific game"""
        url = f"{self.web_base_url}/gamecenter/{game_id}/boxscore"
        return self._make_api_request(url)
    
    def get_game_feed(self, game_id):
        """Get game feed data which includes play-by-play"""
        url = f"{self.web_base_url}/gamecenter/{game_id}/play-by-play"
        return self._make_api_request(url)
    
    def get_player_stats(self, player_id, season=None):
        """Get stats for a specific player"""
        url = f"{self.web_base_url}/player/{player_id}/landing"
        return self._make_api_request(url)
    
    def update_teams(self):
        """Update teams in the database"""
        try:
            # Extract teams from standings data
            teams_data = self.get_teams()
            cursor = self.db_connection.cursor()
            
            team_count = 0
            
            # Process teams data
            for team in teams_data.get('teams', []):
                team_id = team.get('id')
                name = team.get('name')
                abbreviation = team.get('abbrev')
                
                if not team_id or not name:
                    continue
                
                # Check if team exists
                cursor.execute("SELECT 1 FROM teams WHERE team_id = ?", (team_id,))
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO teams (team_id, name, abbreviation, last_updated) VALUES (?, ?, ?, ?)",
                        (team_id, name, abbreviation, datetime.now())
                    )
                else:
                    cursor.execute(
                        "UPDATE teams SET name = ?, abbreviation = ?, last_updated = ? WHERE team_id = ?",
                        (name, abbreviation, datetime.now(), team_id)
                    )
                team_count += 1
            
            self.db_connection.commit()
            logger.info(f"Updated {team_count} teams")
            return True
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error updating teams: {e}")
            return False
    
    def process_completed_games(self, start_date, end_date):
        """Process completed games and update database with results"""
        logger.info(f"Processing completed games from {start_date} to {end_date}")
        
        try:
            schedule = self.get_schedule(start_date, end_date)
            cursor = self.db_connection.cursor()
            games_processed = 0
            
            for game in schedule.get('games', []):
                game_id = game.get('id')
                if not game_id:
                    continue
                    
                season = game.get('season')
                game_date = game.get('gameDate')
                game_state = game.get('gameState')
                
                # Check if both teams exist in our database
                home_team_id = game.get('homeTeam', {}).get('id')
                away_team_id = game.get('awayTeam', {}).get('id')
                
                if not home_team_id or not away_team_id:
                    logger.warning(f"Game {game_id} missing team IDs, skipping")
                    continue
                
                # First, ensure teams exist in our database
                home_team_name = game.get('homeTeam', {}).get('name', {}).get('default', f"Team {home_team_id}")
                away_team_name = game.get('awayTeam', {}).get('name', {}).get('default', f"Team {away_team_id}")
                
                # Insert or update teams
                for team_id, team_name, abbrev in [
                    (home_team_id, home_team_name, game.get('homeTeam', {}).get('abbrev', '')),
                    (away_team_id, away_team_name, game.get('awayTeam', {}).get('abbrev', ''))
                ]:
                    cursor.execute("SELECT 1 FROM teams WHERE team_id = ?", (team_id,))
                    if not cursor.fetchone():
                        cursor.execute(
                            "INSERT INTO teams (team_id, name, abbreviation, last_updated) VALUES (?, ?, ?, ?)",
                            (team_id, team_name, abbrev, datetime.now())
                        )
                
                # Only process final games
                if game_state == 'FINAL' or game_state == 'OFF':
                    # Check if game already processed
                    cursor.execute("SELECT status FROM games WHERE game_id = ?", (game_id,))
                    result = cursor.fetchone()
                    
                    # Skip if already processed as Final
                    if result and result[0] == 'Final':
                        continue
                    
                    # Get scores
                    home_score = game.get('homeTeam', {}).get('score', 0)
                    away_score = game.get('awayTeam', {}).get('score', 0)
                    
                    # Insert/update game
                    cursor.execute(
                        "INSERT OR REPLACE INTO games (game_id, season, game_date, home_team_id, away_team_id, "
                        "status, home_score, away_score, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (game_id, season, game_date, home_team_id, away_team_id, 
                        'Final', home_score, away_score, datetime.now())
                    )
                    
                    # Get detailed game data
                    try:
                        boxscore_data = self.get_game_boxscore(game_id)
                        
                        # Process player stats
                        if boxscore_data:
                            self._process_game_player_stats(game_id, boxscore_data)
                            
                            # Update any predictions with actual results
                            self._update_predictions_results(game_id)
                            
                            games_processed += 1
                        else:
                            logger.warning(f"No boxscore data available for game {game_id}")
                    except Exception as e:
                        logger.error(f"Error processing game {game_id}: {e}")
                        continue
            
            self.db_connection.commit()
            return games_processed
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error in process_completed_games: {e}")
            return 0
    
    def _process_game_player_stats(self, game_id, boxscore_data):
        """Process player stats from a game and update database"""
        cursor = self.db_connection.cursor()
        
        try:
            # Make sure we have boxscore data
            if not boxscore_data:
                logger.warning(f"Empty boxscore data for game {game_id}")
                return False
                
            # Get team IDs
            home_team_id = boxscore_data.get('homeTeam', {}).get('id')
            away_team_id = boxscore_data.get('awayTeam', {}).get('id')
            
            if not home_team_id or not away_team_id:
                # Try to get team IDs from database
                cursor.execute("SELECT home_team_id, away_team_id FROM games WHERE game_id = ?", (game_id,))
                result = cursor.fetchone()
                if result:
                    home_team_id, away_team_id = result
                else:
                    logger.warning(f"Could not determine team IDs for game {game_id}")
                    return False
            
            # Process player data from playerByGameStats
            player_stats = boxscore_data.get('playerByGameStats', {})
            
            # Process home team players
            home_team_data = player_stats.get('homeTeam', {})
            for player_type in ['forwards', 'defense', 'goalies']:
                for player in home_team_data.get(player_type, []):
                    self._process_player_stats(game_id, player, home_team_id, cursor)
            
            # Process away team players
            away_team_data = player_stats.get('awayTeam', {})
            for player_type in ['forwards', 'defense', 'goalies']:
                for player in away_team_data.get(player_type, []):
                    self._process_player_stats(game_id, player, away_team_id, cursor)
            
            return True
        except Exception as e:
            logger.error(f"Error processing player stats for game {game_id}: {e}")
            return False
    
    def _process_player_stats_from_summary(self, game_id, boxscore_data, home_team_id, away_team_id, cursor):
        """Extract player stats from scoring summary"""
        try:
            # Get players who scored goals
            goal_scorers = []
            
            # Go through scoring summary by period
            for period in boxscore_data.get('summary', {}).get('scoring', []):
                for goal in period.get('goals', []):
                    # Get goal scorer info
                    scorer_id = goal.get('scoredBy', {}).get('playerId')
                    if not scorer_id:
                        continue
                    
                    # Determine team
                    team_id = goal.get('teamId')
                    if not team_id:
                        continue
                    
                    # Create minimal player record for scorer
                    first_name = goal.get('scoredBy', {}).get('firstName', {}).get('default', '')
                    last_name = goal.get('scoredBy', {}).get('lastName', {}).get('default', '')
                    name = f"{first_name} {last_name}" if first_name and last_name else f"Player {scorer_id}"
                    
                    # Add to database
                    cursor.execute(
                        "INSERT OR REPLACE INTO players (player_id, name, team_id, position, active, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (scorer_id, name, team_id, "F", 1, datetime.now())
                    )
                    
                    # Add to game stats
                    cursor.execute(
                        "INSERT OR REPLACE INTO player_game_stats "
                        "(player_id, game_id, team_id, position, scored_goal, goals, assists, "
                        "shots, time_on_ice, pp_goals, sh_goals, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (scorer_id, game_id, team_id, "F", 1, 1, 0, 0, 0, 0, 0, datetime.now())
                    )
                    
                    goal_scorers.append(scorer_id)
                    
                    # Process assists
                    for assist in goal.get('assists', []):
                        player_id = assist.get('playerId')
                        if not player_id:
                            continue
                        
                        # Create minimal player record for assist
                        first_name = assist.get('firstName', {}).get('default', '')
                        last_name = assist.get('lastName', {}).get('default', '')
                        name = f"{first_name} {last_name}" if first_name and last_name else f"Player {player_id}"
                        
                        # Add to database
                        cursor.execute(
                            "INSERT OR REPLACE INTO players (player_id, name, team_id, position, active, last_updated) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (player_id, name, team_id, "F", 1, datetime.now())
                        )
                        
                        # Add to game stats
                        cursor.execute(
                            "INSERT OR REPLACE INTO player_game_stats "
                            "(player_id, game_id, team_id, position, scored_goal, goals, assists, "
                            "shots, time_on_ice, pp_goals, sh_goals, last_updated) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (player_id, game_id, team_id, "F", 0, 0, 1, 0, 0, 0, 0, datetime.now())
                        )
            
            return True
        except Exception as e:
            logger.error(f"Error processing summary stats for game {game_id}: {e}")
            return False
    
    def _process_player_stats_from_pbp(self, game_id, pbp_data, home_team_id, away_team_id, cursor):
        """Extract player stats from play-by-play data"""
        try:
            # Get all players from rosterSpots
            roster_spots = pbp_data.get('rosterSpots', {})
            
            # Process each player
            for player_id, player_data in roster_spots.items():
                team_id = player_data.get('teamId')
                if not team_id:
                    continue
                
                # Get player name and position
                first_name = player_data.get('firstName', {}).get('default', '')
                last_name = player_data.get('lastName', {}).get('default', '')
                name = f"{first_name} {last_name}" if first_name and last_name else f"Player {player_id}"
                position = player_data.get('positionCode', '')
                
                # Add player to database
                cursor.execute(
                    "INSERT OR REPLACE INTO players (player_id, name, team_id, position, active, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (player_id, name, team_id, position, 1, datetime.now())
                )
                
                # Default values
                scored_goal = 0
                goals = 0
                assists = 0
                
                # Look for scoring plays by this player
                for play in pbp_data.get('plays', []):
                    if play.get('typeCode') == 'GOAL':
                        # Check if this player scored
                        if play.get('details', {}).get('scoringPlayerId') == player_id:
                            scored_goal = 1
                            goals += 1
                        
                        # Check for assists
                        for assist_key in ['assist1PlayerId', 'assist2PlayerId']:
                            if play.get('details', {}).get(assist_key) == player_id:
                                assists += 1
                
                # Add to game stats
                cursor.execute(
                    "INSERT OR REPLACE INTO player_game_stats "
                    "(player_id, game_id, team_id, position, scored_goal, goals, assists, "
                    "shots, time_on_ice, pp_goals, sh_goals, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (player_id, game_id, team_id, position, scored_goal, goals, assists, 0, 0, 0, 0, datetime.now())
                )
            
            return True
        except Exception as e:
            logger.error(f"Error processing play-by-play stats for game {game_id}: {e}")
            return False
    
    def _process_player_stats(self, game_id, player, team_id, cursor):
        """Process stats for a single player"""
        try:
            player_id = player.get('playerId')
            if not player_id:
                return
                
            # Get player name from the new structure
            player_name = player.get('name', {}).get('default', f"Player {player_id}")
            position = player.get('position')
            
            # Update player in database
            cursor.execute(
                "INSERT OR REPLACE INTO players (player_id, name, team_id, position, active, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, player_name, team_id, position, 1, datetime.now())
            )
            
            # For skaters
            if position != 'G':
                goals = player.get('goals', 0)
                assists = player.get('assists', 0)
                shots = player.get('sog', 0)
                
                # Convert time on ice to seconds
                time_on_ice_str = player.get('toi', '0:00')
                time_parts = time_on_ice_str.split(':')
                time_on_ice = int(time_parts[0]) * 60 + int(time_parts[1]) if len(time_parts) >= 2 else 0
                
                # Power play and shorthanded goals
                pp_goals = player.get('powerPlayGoals', 0)
                sh_goals = 0  # May need to look for this in the API
                
                # Update player_game_stats
                cursor.execute(
                    "INSERT OR REPLACE INTO player_game_stats "
                    "(player_id, game_id, team_id, position, scored_goal, goals, assists, "
                    "shots, time_on_ice, pp_goals, sh_goals, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (player_id, game_id, team_id, position, goals > 0, goals, assists,
                    shots, time_on_ice, pp_goals, sh_goals, datetime.now())
                )
            else:
                # For goalies, just record that they played
                cursor.execute(
                    "INSERT OR REPLACE INTO player_game_stats "
                    "(player_id, game_id, team_id, position, scored_goal, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (player_id, game_id, team_id, position, 0, datetime.now())
                )
        except Exception as e:
            logger.error(f"Error processing stats for player {player.get('playerId')}: {e}")

    def _update_predictions_results(self, game_id):
        """Update predictions with actual results"""
        cursor = self.db_connection.cursor()
        
        try:
            # Get all players who scored in this game
            cursor.execute(
                "SELECT player_id FROM player_game_stats WHERE game_id = ? AND scored_goal = 1",
                (game_id,)
            )
            scorers = set(row[0] for row in cursor.fetchall())
            
            # Update all predictions for this game
            cursor.execute(
                "SELECT player_id, prediction_id FROM predictions WHERE game_id = ? AND scored IS NULL", 
                (game_id,)
            )
            
            for player_id, prediction_id in cursor.fetchall():
                scored = player_id in scorers
                cursor.execute(
                    "UPDATE predictions SET scored = ?, last_updated = ? WHERE prediction_id = ?",
                    (scored, datetime.now(), prediction_id)
                )
            
            self.db_connection.commit()
            return True
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error updating prediction results for game {game_id}: {e}")
            return False
    
    def get_upcoming_games(self, start_date, end_date):
        """Get list of upcoming games with improved team name handling"""
        try:
            schedule = self.get_schedule(start_date, end_date)
            cursor = self.db_connection.cursor()
            upcoming_games = []
            
            for game in schedule.get('games', []):
                game_id = game.get('id')
                game_date = game.get('gameDate', '').split('T')[0] if 'gameDate' in game else game.get('startTimeUTC', '').split('T')[0]
                game_state = game.get('gameState')
                
                # Only include upcoming games
                if game_state != 'FINAL' and game_state != 'OFF':
                    home_team_id = game.get('homeTeam', {}).get('id')
                    away_team_id = game.get('awayTeam', {}).get('id')
                    
                    # Ensure team IDs are valid
                    if not home_team_id or not away_team_id:
                        logger.warning(f"Invalid team IDs for game {game_id}")
                        continue
                    
                    # Get team names using our helper method
                    home_team = self._get_team_name(home_team_id)
                    away_team = self._get_team_name(away_team_id)
                    
                    game_info = {
                        'game_id': game_id,
                        'date': game_date,
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'status': game_state
                    }
                    
                    # Insert or update in database
                    cursor.execute(
                        "INSERT OR REPLACE INTO games "
                        "(game_id, season, game_date, home_team_id, away_team_id, status, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (game_id, game.get('season', ''), game_date, home_team_id, away_team_id, 
                        game_state, datetime.now())
                    )
                    
                    # Also update team names in the teams table
                    self._update_team_in_database(home_team_id, home_team)
                    self._update_team_in_database(away_team_id, away_team)
                    
                    upcoming_games.append(game_info)
            
            self.db_connection.commit()
            return upcoming_games
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error getting upcoming games: {e}")
            return []

    def _update_team_in_database(self, team_id, team_name):
        """Helper method to update team name in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO teams (team_id, name, last_updated) "
                "VALUES (?, ?, ?)",
                (team_id, team_name, datetime.now())
            )
        except Exception as e:
            logger.error(f"Error updating team in database: {e}")

    def _get_team_name(self, team_id, default_name=None):
        """Get team name from NHL_TEAM_NAMES mapping or database with fallback"""
        try:
            # First try to get from our mapping
            from config import NHL_TEAM_NAMES
            if team_id in NHL_TEAM_NAMES:
                team_name = NHL_TEAM_NAMES[team_id]
                
                # Update database with this name
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "UPDATE teams SET name = ? WHERE team_id = ?",
                    (team_name, team_id)
                )
                self.db_connection.commit()
                return team_name
            
            # If not in mapping, try database
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team_id,))
            result = cursor.fetchone()
            
            if result and result[0] and not result[0].startswith('Team '):
                return result[0]
            
            # Return a fallback name if nothing else worked
            if default_name and not default_name.startswith('Team '):
                return default_name
            
            # Last resort
            return f"NHL Team {team_id}"
        
        except Exception as e:
            logger.error(f"Error getting team name for ID {team_id}: {e}")
            return default_name if default_name else f"NHL Team {team_id}"