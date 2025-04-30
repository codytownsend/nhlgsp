import sqlite3
import json
import logging
import argparse
from datetime import datetime, timedelta
from nhl_data_collector import NHLDataCollector
from goal_scorer_model import GoalScorerModel
from database import create_database, get_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nhl_predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the system - create database and tables"""
    logger.info("Initializing NHL prediction system")
    conn = create_database()
    return conn

def update_data(conn, start_date=None, end_date=None):
    """Update data from NHL API"""
    collector = NHLDataCollector(conn)
    
    # Set date range if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Updating data from {start_date} to {end_date}")
    
    # Update teams
    logger.info("Updating teams...")
    collector.update_teams()
    
    # Process completed games
    logger.info("Processing completed games...")
    games_processed = collector.process_completed_games(start_date, end_date)
    logger.info(f"Processed {games_processed} games")
    
    return True

def train_model(conn):
    """Train the goal scorer prediction model"""
    logger.info("Training goal scorer model")
    model = GoalScorerModel(conn)
    success = model.train()
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.warning("Model training may not have completed successfully")
    
    return model

def make_predictions(conn, start_date=None, end_date=None):
    """Make predictions for upcoming games"""
    collector = NHLDataCollector(conn)
    model = GoalScorerModel(conn)
    
    # Set date range if not provided
    if start_date is None:
        start_date = datetime.now().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"Making predictions for games from {start_date} to {end_date}")
    
    # Get upcoming games
    upcoming_games = collector.get_upcoming_games(start_date, end_date)
    
    if not upcoming_games:
        logger.warning(f"No upcoming games found for date range {start_date} to {end_date}")
        return []
    
    all_predictions = []
    
    # Make predictions for each game
    for game in upcoming_games:
        game_id = game['game_id']
        logger.info(f"Generating predictions for {game['home_team']} vs {game['away_team']} (ID: {game_id})")
        
        predictions = model.predict_game(game_id)
        if predictions:
            # Format for display
            game_predictions = {
                'game_id': game_id,
                'date': game['date'],
                'matchup': f"{game['home_team']} vs {game['away_team']}",
                'predictions': predictions
            }
            all_predictions.append(game_predictions)
    
    return all_predictions

def display_predictions(predictions):
    """Display predictions in a readable format"""
    if not predictions:
        print("No predictions available.")
        return
    
    for game in predictions:
        print(f"\n=== {game['matchup']} ({game['date']}) ===")
        print("Top 10 predicted goal scorers:")
        
        for i, pred in enumerate(game['predictions'][:10], 1):
            prob_pct = pred['probability'] * 100
            print(f"{i}. {pred['name']} ({pred['team']}, {pred['position']}) - {prob_pct:.1f}%")
    
    print("\nNote: Probabilities represent a player's chance of scoring at least one goal in this game.")

def export_predictions_to_json(predictions, filename=None):
    """Export predictions to a JSON file"""
    if not predictions:
        logger.warning("No predictions to export")
        return False
    
    if filename is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"predictions_{now}.json"
    
    try:
        # Convert to serializable format
        serializable_predictions = []
        for game in predictions:
            game_data = {
                'game_id': game['game_id'],
                'date': game['date'],
                'matchup': game['matchup'],
                'predictions': []
            }
            
            for pred in game['predictions']:
                # Remove non-serializable features to avoid JSON encoding issues
                if 'features' in pred:
                    features = pred['features']
                else:
                    features = {}
                
                pred_data = {
                    'player_id': pred['player_id'],
                    'name': pred['name'],
                    'team': pred['team'],
                    'position': pred['position'],
                    'probability': float(pred['probability']),
                    'features': features
                }
                game_data['predictions'].append(pred_data)
            
            serializable_predictions.append(game_data)
        
        with open(filename, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
        
        logger.info(f"Predictions exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        return False

def evaluate_predictions(conn, days_ago=1):
    """Evaluate prediction accuracy for games from a specified number of days ago"""
    logger.info(f"Evaluating predictions from {days_ago} days ago")
    
    target_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    try:
        cursor = conn.cursor()
        
        # Get games from target date
        cursor.execute(
            "SELECT g.game_id, home.name as home_team, away.name as away_team "
            "FROM games g "
            "JOIN teams home ON g.home_team_id = home.team_id "
            "JOIN teams away ON g.away_team_id = away.team_id "
            "WHERE DATE(g.game_date) = ? AND g.status = 'Final'",
            (target_date,)
        )
        games = cursor.fetchall()
        
        if not games:
            logger.warning(f"No completed games found for {target_date}")
            return
        
        print(f"\n=== Prediction Evaluation for {target_date} ===")
        
        overall_correct = 0
        overall_total = 0
        
        for game_id, home_team, away_team in games:
            # Get top 10 predictions for this game
            cursor.execute(
                "SELECT p.player_id, pl.name, pr.goal_probability, pr.scored "
                "FROM predictions pr "
                "JOIN players pl ON pr.player_id = pl.player_id "
                "WHERE pr.game_id = ? "
                "ORDER BY pr.goal_probability DESC LIMIT 10",
                (game_id,)
            )
            predictions = cursor.fetchall()
            
            if not predictions:
                logger.warning(f"No predictions found for game {game_id}")
                continue
            
            # Get actual goal scorers
            cursor.execute(
                "SELECT p.name FROM player_game_stats pgs "
                "JOIN players p ON pgs.player_id = p.player_id "
                "WHERE pgs.game_id = ? AND pgs.scored_goal = 1",
                (game_id,)
            )
            scorers = [row[0] for row in cursor.fetchall()]
            
            print(f"\n{home_team} vs {away_team}")
            print(f"Actual goal scorers: {', '.join(scorers) if scorers else 'None'}")
            
            correct_predictions = 0
            for _, name, prob, scored in predictions:
                status = "✓" if scored else "✗"
                print(f"{status} {name} - {prob*100:.1f}%")
                if scored:
                    correct_predictions += 1
            
            print(f"Accuracy: {correct_predictions}/{len(predictions)} predictions correct")
            
            overall_correct += correct_predictions
            overall_total += len(predictions)
        
        if overall_total > 0:
            print(f"\nOverall accuracy: {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.1f}%)")
        
        # Store evaluation metrics
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS prediction_evaluation ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "evaluation_date TIMESTAMP, "
            "target_date DATE, "
            "correct_predictions INTEGER, "
            "total_predictions INTEGER, "
            "accuracy REAL)"
        )
        
        cursor.execute(
            "INSERT INTO prediction_evaluation "
            "(evaluation_date, target_date, correct_predictions, total_predictions, accuracy) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(), target_date, overall_correct, overall_total, 
             overall_correct/overall_total if overall_total > 0 else 0)
        )
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error evaluating predictions: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='NHL Goal Scorer Prediction System')
    parser.add_argument('--update', action='store_true', help='Update data from NHL API')
    parser.add_argument('--train', action='store_true', help='Train the prediction model')
    parser.add_argument('--predict', action='store_true', help='Make predictions for upcoming games')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictions from previous days')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--export', action='store_true', help='Export predictions to JSON')
    parser.add_argument('--days-ago', type=int, default=1, help='Days ago for evaluation')
    
    args = parser.parse_args()
    
    # Default behavior if no arguments provided
    if not (args.update or args.train or args.predict or args.evaluate):
        args.update = True
        args.train = True
        args.predict = True
    
    # Initialize system
    conn = initialize_system()
    
    try:
        # Update data
        if args.update:
            update_data(conn, args.start_date, args.end_date)
        
        # Train model
        if args.train:
            train_model(conn)
        
        # Make predictions
        if args.predict:
            predictions = make_predictions(conn, args.start_date, args.end_date)
            display_predictions(predictions)
            
            if args.export and predictions:
                export_predictions_to_json(predictions)
        
        # Evaluate predictions
        if args.evaluate:
            evaluate_predictions(conn, args.days_ago)
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
    finally:
        # Close database connection
        conn.close()

if __name__ == "__main__":
    main()