import requests
import json
import pandas as pd
from datetime import datetime, timedelta

class NHLDataCollector:
    def __init__(self, db_connection):
        self.base_url = "https://statsapi.web.nhl.com/api/v1"
        self.db_connection = db_connection
        
    def get_schedule(self, start_date, end_date):
        """Get NHL schedule between two dates"""
        url = f"{self.base_url}/schedule?startDate={start_date}&endDate={end_date}"
        response = requests.get(url)
        return response.json()
    
    def get_game_data(self, game_id):
        """Get detailed data for a specific game"""
        url = f"{self.base_url}/game/{game_id}/feed/live"
        response = requests.get(url)
        return response.json()
    
    def get_player_stats(self, player_id):
        """Get season stats for a specific player"""
        url = f"{self.base_url}/people/{player_id}/stats?stats=statsSingleSeason"
        response = requests.get(url)
        return response.json()
    
    def get_teams(self):
        """Get all NHL teams"""
        url = f"{self.base_url}/teams"
        response = requests.get(url)
        return response.json()