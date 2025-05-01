# NHL Goal Scorer Prediction System

A data-driven system for predicting which NHL players are most likely to score goals in upcoming games.

## Features

- NHL API integration for collecting game and player statistics
- Player statistics tracking and analysis
- Goal prediction model using machine learning
- Prediction accuracy evaluation
- Command-line interface for easy operation

## Requirements

- Python 3.7 or higher
- Required packages: `requests`, `pandas`, `numpy`, `scikit-learn`, `joblib`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nhl-goal-scorer-prediction.git
   cd nhl-goal-scorer-prediction
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guide

### Step 1: Collect Historical Data

First, collect data from previous seasons to build a robust training dataset:

```bash
# Collect data from the 2021-2022 NHL season
python3 main.py --historical --season 20212022

# Add data from the 2022-2023 season
python3 main.py --historical --season 20222023

# Add data from the 2023-2024 season
python3 main.py --historical --season 20232024
```

Adding multiple seasons creates a more comprehensive dataset for better predictions.

### Step 2: Train the Prediction Model

After collecting historical data, train the prediction model:

```bash
# Train using existing data
python3 main.py --train

# Force a complete model rebuild
python3 main.py --train --force-retrain
```

### Step 3: Update with Recent Data

Keep your database current by updating with recent games:

```bash
# Update with data from the last 3 days (default)
python3 main.py --update

# Update with data from a specific date range
python3 main.py --update --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

### Step 4: Generate Predictions

Generate predictions for upcoming games:

```bash
# Predict for today and tomorrow (default)
python3 main.py --predict

# Export predictions to a JSON file
python3 main.py --predict --export

# Predict for a specific date range
python3 main.py --predict --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

### Step 5: Evaluate Prediction Accuracy

After games have been played, evaluate your model's performance:

```bash
# Evaluate predictions from yesterday (default)
python3 main.py --evaluate

# Evaluate predictions from N days ago
python3 main.py --evaluate --days-ago N
```

### All-in-One Command

For regular use, you can run the entire pipeline with one command:

```bash
python3 main.py --update --train --predict --export
```

This will:
1. Update your database with recent games
2. Train/update the prediction model  
3. Generate and display predictions
4. Export predictions to a JSON file

## Recommended Workflow

For the best results, we recommend:

1. Initially collect 2-3 seasons of historical data
2. Train your model with this comprehensive dataset
3. Run daily updates during the NHL season to:
   - Update with recent game results
   - Make new predictions for upcoming games  
   - Evaluate previous predictions

## Command Reference

| Flag | Description |
|------|-------------|
| `--historical` | Collect historical data (pair with `--season`) |
| `--season` | Specify NHL season in YYYYYYYY format (e.g., 20232024) |  
| `--update` | Update database with recent games |
| `--train` | Train the prediction model |
| `--force-retrain` | Force rebuild of model even if one exists |
| `--predict` | Make predictions for upcoming games |
| `--evaluate` | Evaluate prediction accuracy |
| `--start-date` | Specify start date (YYYY-MM-DD) |
| `--end-date` | Specify end date (YYYY-MM-DD) |
| `--export` | Export predictions to JSON file |
| `--days-ago` | Days ago for evaluation (default: 1) |

## Output Files

- Database: `nhl_predictions.db`
- Model: `models/goal_scorer_model.joblib`
- Predictions: `output/predictions_YYYYMMDD_HHMMSS.json`
- Logs: `nhl_predictions.log`, `nhl_data_collector.log`, `goal_scorer_model.log`