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
        self.base_url = "https://statsapi.web.nhl.com/api/v1"
        self.db_connection = db_connection
        
    def _make_api_request(self, endpoint, params=None, max_retries=3, retry_delay=5):
        """Make an API request with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        retries = 0
        
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
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
                    raise
    
    def get_teams(self):
        """Get all NHL teams"""
        logger.info("Fetching teams data")
        return self._make_api_request("teams")
        
    def get_schedule(self, start_date, end_date):
        """Get NHL schedule between two dates"""
        logger.info(f"Fetching schedule from {start_date} to {end_date}")
        return self._make_api_request("schedule", {
            "startDate": start_date, 
            "endDate": end_date
        })
    
    def get_game_data(self, game_id):
        """Get detailed data for a specific game"""
        logger.info(f"Fetching data for game {game_id}")
        return self._make_api_request(f"game/{game_id}/feed/live")
    
    def get_player_stats(self, player_id, season=None):
        """Get season stats for a specific player"""
        logger.info(f"Fetching stats for player {player_id}")
        params = {"stats": "statsSingleSeason"}
        if season:
            params["season"] = season
        return self._make_api_request(f"people/{player_id}/stats", params)
    
    def update_teams(self):
        """Update teams in the database"""
        logger.info("Updating teams in database")
        try:
            teams_data = self.get_teams()
            cursor = self.db_connection.cursor()
            
            for team in teams_data['teams']:
                team_id = team['id']
                name = team['name']
                abbreviation = team.get('abbreviation', '')
                
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
            
            self.db_connection.commit()
            logger.info(f"Updated {len(teams_data['teams'])} teams")
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
            
            for date in schedule.get('dates', []):
                for game in date.get('games', []):
                    game_id = game['gamePk']
                    game_date = date['date']
                    season = game['season']
                    home_team_id = game['teams']['home']['team']['id']
                    away_team_id = game['teams']['away']['team']['id']
                    status = game['status']['detailedState']
                    
                    # Only process final games
                    if status == 'Final':
                        # Check if game already processed
                        cursor.execute("SELECT status FROM games WHERE game_id = ?", (game_id,))
                        result = cursor.fetchone()
                        
                        # Skip if already processed as Final
                        if result and result[0] == 'Final':
                            logger.info(f"Game {game_id} already processed, skipping")
                            continue
                        
                        # Get scores
                        home_score = game['teams']['home']['score']
                        away_score = game['teams']['away']['score']
                        
                        # Insert/update game
                        cursor.execute(
                            "INSERT OR REPLACE INTO games (game_id, season, game_date, home_team_id, away_team_id, "
                            "status, home_score, away_score, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (game_id, season, game_date, home_team_id, away_team_id, 
                             status, home_score, away_score, datetime.now())
                        )
                        
                        # Get detailed game data
                        try:
                            game_data = self.get_game_data(game_id)
                            
                            # Process player stats
                            self._process_game_player_stats(game_id, game_data)
                            
                            # Update any predictions with actual results
                            self._update_predictions_results(game_id, game_data)
                            
                            games_processed += 1
                            logger.info(f"Processed game {game_id}: {home_score}-{away_score}")
                        except Exception as e:
                            logger.error(f"Error processing game {game_id}: {e}")
                            continue
            
            self.db_connection.commit()
            logger.info(f"Completed processing {games_processed} games")
            return games_processed
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error in process_completed_games: {e}")
            return 0
    
    def _process_game_player_stats(self, game_id, game_data):
        """Process player stats from a game and update database"""
        cursor = self.db_connection.cursor()
        
        try:
            # Extract boxscore data which has player stats
            boxscore = game_data['liveData']['boxscore']
            
            # Process each team's players
            for team_type in ['home', 'away']:
                team = game_data['gameData']['teams'][team_type]
                team_id = team['id']
                
                # Process all players in the boxscore
                for player_id_str, player_stats in boxscore['teams'][team_type]['players'].items():
                    # Player IDs in the API have a prefix 'ID'
                    player_id = int(player_id_str.replace('ID', ''))
                    
                    # Basic player info
                    player_info = player_stats['person']
                    player_name = player_info['fullName']
                    position = player_stats.get('position', {}).get('abbreviation', '')
                    
                    # Update player in database
                    cursor.execute(
                        "INSERT OR REPLACE INTO players (player_id, name, team_id, position, active, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (player_id, player_name, team_id, position, 1, datetime.now())
                    )
                    
                    # Extract stats if player played
                    if 'stats' in player_stats and 'skaterStats' in player_stats['stats']:
                        skater_stats = player_stats['stats']['skaterStats']
                        
                        goals = skater_stats.get('goals', 0)
                        assists = skater_stats.get('assists', 0)
                        shots = skater_stats.get('shots', 0)
                        time_on_ice_str = skater_stats.get('timeOnIce', '0:00')
                        
                        # Convert time on ice to seconds
                        time_parts = time_on_ice_str.split(':')
                        time_on_ice = int(time_parts[0]) * 60 + int(time_parts[1])
                        
                        # Power play and shorthanded goals
                        pp_goals = skater_stats.get('powerPlayGoals', 0)
                        sh_goals = skater_stats.get('shortHandedGoals', 0)
                        
                        # Update player_game_stats
                        cursor.execute(
                            "INSERT OR REPLACE INTO player_game_stats "
                            "(player_id, game_id, team_id, position, scored_goal, goals, assists, "
                            "shots, time_on_ice, pp_goals, sh_goals, last_updated) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (player_id, game_id, team_id, position, goals > 0, goals, assists,
                             shots, time_on_ice, pp_goals, sh_goals, datetime.now())
                        )
                    
                    # For goalies, different stats structure
                    elif 'stats' in player_stats and 'goalieStats' in player_stats['stats']:
                        # Insert basic record for the goalie
                        cursor.execute(
                            "INSERT OR REPLACE INTO player_game_stats "
                            "(player_id, game_id, team_id, position, scored_goal, last_updated) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (player_id, game_id, team_id, position, 0, datetime.now())
                        )
            
            return True
        except Exception as e:
            logger.error(f"Error processing player stats for game {game_id}: {e}")
            return False
    
    def _update_predictions_results(self, game_id, game_data):
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
            logger.info(f"Updated prediction results for game {game_id}")
            return True
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error updating prediction results for game {game_id}: {e}")
            return False
    
    def get_upcoming_games(self, start_date, end_date):
        """Get list of upcoming games"""
        logger.info(f"Finding upcoming games from {start_date} to {end_date}")
        try:
            schedule = self.get_schedule(start_date, end_date)
            cursor = self.db_connection.cursor()
            upcoming_games = []
            
            for date in schedule.get('dates', []):
                for game in date.get('games', []):
                    game_id = game['gamePk']
                    game_date = date['date']
                    status = game['status']['abstractGameState']
                    
                    if status != 'Final':
                        home_team_id = game['teams']['home']['team']['id']
                        away_team_id = game['teams']['away']['team']['id']
                        
                        # Get team names
                        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (home_team_id,))
                        home_team_result = cursor.fetchone()
                        home_team = home_team_result[0] if home_team_result else "Unknown"
                        
                        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (away_team_id,))
                        away_team_result = cursor.fetchone()
                        away_team = away_team_result[0] if away_team_result else "Unknown"
                        
                        game_info = {
                            'game_id': game_id,
                            'date': game_date,
                            'home_team_id': home_team_id,
                            'away_team_id': away_team_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'status': status
                        }
                        
                        # Insert or update in database
                        cursor.execute(
                            "INSERT OR REPLACE INTO games "
                            "(game_id, season, game_date, home_team_id, away_team_id, status, last_updated) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (game_id, game.get('season', ''), game_date, home_team_id, away_team_id, 
                             status, datetime.now())
                        )
                        
                        upcoming_games.append(game_info)
            
            self.db_connection.commit()
            return upcoming_games
        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error getting upcoming games: {e}")
            return []