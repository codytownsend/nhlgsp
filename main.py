import sqlite3
from datetime import datetime, timedelta
from nhl_data_collector import NHLDataCollector
from goal_scorer_model import GoalScorerModel
from database import create_database, get_connection

def main():
    # Initialize database
    conn = create_database()
    
    # Initialize data collector and model
    collector = NHLDataCollector(conn)
    model = GoalScorerModel(conn)
    
    # Get today's date and yesterday's date
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Collect data
    print("Collecting schedule data...")
    schedule = collector.get_schedule(yesterday, today)
    
    # Process yesterday's games for training data
    print("Processing completed games...")
    # TODO: Implement code to process completed games and update database
    
    # Train model
    print("Training model...")
    model.train()
    
    # Make predictions for upcoming games
    print("Making predictions for today's games...")
    # TODO: Implement code to make predictions
    
    # Close database connection
    conn.close()
    
if __name__ == "__main__":
    main()