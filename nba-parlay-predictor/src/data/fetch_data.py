"""
Functions to fetch NBA and betting odds data
"""
import pandas as pd
import numpy as np
import os
import config

def load_nba_data(season='2023-24'):
    """
    Load NBA game data
    
    For a one-day project, we'll load from local CSV files
    In a full implementation, you'd use the NBA API
    """
    file_path = f"{config.DATA_RAW_DIR}/nba_games_{season.replace('-', '_')}.csv"
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # In a real implementation, you'd use:
        # from nba_api.stats.endpoints import leaguegamefinder
        # gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        # return gamefinder.get_data_frames()[0]
        raise FileNotFoundError(f"NBA data file not found: {file_path}")

def load_odds_data():
    """
    Load betting odds data
    
    For a one-day project, we'll load from local CSV files
    In a full implementation, you'd use a betting API or scraper
    """
    file_path = f"{config.DATA_RAW_DIR}/betting_odds.csv"
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Betting odds file not found: {file_path}")

def generate_synthetic_data(n_teams=30, n_games=500):
    """
    Generate synthetic NBA game and odds data for testing
    """
    # Generate synthetic NBA data
    teams = [f"Team_{i}" for i in range(1, n_teams+1)]
    
    games = []
    for _ in range(n_games):
        # Select random teams
        home_team, away_team = np.random.choice(teams, size=2, replace=False)
        
        # Generate random stats
        home_pts = np.random.randint(85, 130)
        away_pts = np.random.randint(85, 130)
        
        game = {
            'GAME_ID': np.random.randint(10000, 99999),
            'GAME_DATE': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 150)),
            'TEAM_NAME': home_team,
            'MATCHUP': f"{home_team} vs. {away_team}",
            'WL': 'W' if home_pts > away_pts else 'L',
            'PTS': home_pts,
            'FG_PCT': np.random.uniform(0.4, 0.55),
            'FT_PCT': np.random.uniform(0.7, 0.9),
            'REB': np.random.randint(30, 55),
            'AST': np.random.randint(15, 35),
            'STL': np.random.randint(5, 15),
            'BLK': np.random.randint(2, 10),
            'TOV': np.random.randint(8, 20),
            'PLUS_MINUS': home_pts - away_pts
        }
        games.append(game)
        
        # Add corresponding away game
        away_game = {
            'GAME_ID': game['GAME_ID'],
            'GAME_DATE': game['GAME_DATE'],
            'TEAM_NAME': away_team,
            'MATCHUP': f"{away_team} @ {home_team}",
            'WL': 'W' if away_pts > home_pts else 'L',
            'PTS': away_pts,
            'FG_PCT': np.random.uniform(0.4, 0.55),
            'FT_PCT': np.random.uniform(0.7, 0.9),
            'REB': np.random.randint(30, 55),
            'AST': np.random.randint(15, 35),
            'STL': np.random.randint(5, 15),
            'BLK': np.random.randint(2, 10),
            'TOV': np.random.randint(8, 20),
            'PLUS_MINUS': away_pts - home_pts
        }
        games.append(away_game)
    
    games_df = pd.DataFrame(games)
    
    # Generate synthetic odds data
    home_games = games_df[games_df['MATCHUP'].str.contains('vs.')].copy()
    
    odds_data = []
    for _, game in home_games.iterrows():
        home_team = game['TEAM_NAME']
        away_team = game['MATCHUP'].split('vs. ')[1]
        game_date = game['GAME_DATE']
        
        # Generate synthetic odds
        home_win = game['WL'] == 'W'
        
        if home_win:
            home_odds = np.random.uniform(-130, -110)
            away_odds = np.random.uniform(100, 120)
            spread = np.random.uniform(-4.5, -1.5)
        else:
            home_odds = np.random.uniform(100, 120)
            away_odds = np.random.uniform(-130, -110)
            spread = np.random.uniform(1.5, 4.5)
            
        odds_data.append({
            'GAME_DATE': game_date,
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'HOME_ODDS': home_odds,
            'AWAY_ODDS': away_odds,
            'SPREAD': spread
        })
    
    odds_df = pd.DataFrame(odds_data)
    
    # Save synthetic data
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    games_df.to_csv(f"{config.DATA_RAW_DIR}/synthetic_nba_games.csv", index=False)
    odds_df.to_csv(f"{config.DATA_RAW_DIR}/synthetic_betting_odds.csv", index=False)
    
    return games_df, odds_df