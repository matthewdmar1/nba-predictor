"""
Functions to fetch real NBA and betting odds data
"""
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoresummaryv2, boxscoretraditionalv2
from nba_api.stats.static import teams
import config

def get_nba_teams():
    """
    Get a dictionary of NBA teams with IDs
    
    Returns:
    --------
    team_dict : dict
        Dictionary mapping team names to team IDs
    """
    # Get all NBA teams
    nba_teams = teams.get_teams()
    
    # Create dictionary mapping team names to IDs
    team_dict = {team['full_name']: team['id'] for team in nba_teams}
    
    return team_dict

def load_nba_data(season='2023-24', max_games=None):
    """
    Load NBA game data from the NBA API
    
    Parameters:
    -----------
    season : str
        NBA season in format 'YYYY-YY'
    max_games : int, optional
        Maximum number of games to load (for testing purposes)
        
    Returns:
    --------
    games_df : pandas DataFrame
        DataFrame with NBA game data
    """
    # Check if cached data exists
    cache_file = f"{config.DATA_RAW_DIR}/nba_games_{season.replace('-', '_')}.csv"
    
    if os.path.exists(cache_file):
        print(f"Loading cached NBA data from {cache_file}")
        return pd.read_csv(cache_file)
    
    print(f"Fetching NBA game data for season {season}...")
    
    # Use NBA API to get game data
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00',  # NBA league ID
        season_type_nullable='Regular Season'
    )
    
    # Convert to DataFrame
    games_df = gamefinder.get_data_frames()[0]
    
    # Limit number of games if specified
    if max_games is not None:
        games_df = games_df.head(max_games)
    
    # Add detailed stats for each game
    # WARNING: This makes many API calls and may hit rate limits
    if not max_games:
        games_df = enrich_game_data(games_df)
    
    # Save to cache file
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    games_df.to_csv(cache_file, index=False)
    print(f"Saved NBA data to {cache_file}")
    
    return games_df

def enrich_game_data(games_df, sample_size=None):
    """
    Enrich game data with detailed stats from box scores
    
    Parameters:
    -----------
    games_df : pandas DataFrame
        DataFrame with basic game data
    sample_size : int, optional
        Number of games to process (for testing purposes)
        
    Returns:
    --------
    enriched_df : pandas DataFrame
        DataFrame with additional game statistics
    """
    print("Enriching game data with detailed statistics...")
    
    # Get unique game IDs
    game_ids = games_df['GAME_ID'].unique()
    
    # Limit to sample size if specified
    if sample_size:
        game_ids = game_ids[:sample_size]
    
    # Initialize list to store additional data
    additional_data = []
    
    # Process each game
    for i, game_id in enumerate(game_ids):
        if i % 10 == 0:
            print(f"Processing game {i+1} of {len(game_ids)}...")
        
        try:
            # Get box score summary
            box_summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
            game_info = box_summary.game_info.get_data_frame()
            
            # Get traditional box score
            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = box_score.player_stats.get_data_frame()
            
            # Calculate team stats by aggregating player stats
            team_stats = player_stats.groupby('TEAM_ID').agg({
                'FG_PCT': 'mean',
                'FT_PCT': 'mean',
                'REB': 'sum',
                'AST': 'sum',
                'STL': 'sum',
                'BLK': 'sum',
                'TOV': 'sum',
                'PLUS_MINUS': 'sum'
            }).reset_index()
            
            # Add to additional data
            for _, team_row in team_stats.iterrows():
                team_games = games_df[
                    (games_df['GAME_ID'] == game_id) & 
                    (games_df['TEAM_ID'] == team_row['TEAM_ID'])
                ]
                
                if len(team_games) > 0:
                    game_record = team_games.iloc[0].to_dict()
                    game_record.update({
                        'FG_PCT': team_row['FG_PCT'],
                        'FT_PCT': team_row['FT_PCT'],
                        'REB': team_row['REB'],
                        'AST': team_row['AST'],
                        'STL': team_row['STL'],
                        'BLK': team_row['BLK'],
                        'TOV': team_row['TOV']
                    })
                    additional_data.append(game_record)
            
            # Respect API rate limits
            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
    
    # Convert to DataFrame
    if additional_data:
        enriched_df = pd.DataFrame(additional_data)
        return enriched_df
    else:
        return games_df

def load_odds_data(season='2023-24'):
    """
    Load betting odds data from the API
    
    For real implementation, consider using a sports betting API like:
    - The Odds API
    - SportsData.io
    - BetConstruct
    
    Parameters:
    -----------
    season : str
        NBA season in format 'YYYY-YY'
        
    Returns:
    --------
    odds_df : pandas DataFrame
        DataFrame with betting odds data
    """
    # Check if cached data exists
    cache_file = f"{config.DATA_RAW_DIR}/betting_odds_{season.replace('-', '_')}.csv"
    
    if os.path.exists(cache_file):
        print(f"Loading cached betting odds from {cache_file}")
        return pd.read_csv(cache_file)
    
    # Try to fetch odds data from a free API
    # For demonstration purposes, we're using a free NBA odds API
    # In a production environment, you'd likely use a paid service with better data
    
    try:
        # Get data for the 2023-2024 season
        # Note: Free APIs often have limited historical data
        season_year = int(season.split('-')[0])
        
        # Define date range for the season
        start_date = f"{season_year}-10-01" 
        end_date = f"{season_year+1}-06-30"
        
        print(f"Fetching betting odds data from {start_date} to {end_date}...")
        
        # For demo purposes, try to get odds data from a public API
        # Replace with your preferred odds provider in production
        url = "https://odds.p.rapidapi.com/v4/sports/basketball_nba/odds"
        
        querystring = {
            "regions":"us",
            "oddsFormat":"american",
            "markets":"h2h,spreads",
            "dateFormat":"iso"
        }
        
        # You would need to sign up for an API key
        headers = {
            "X-RapidAPI-Key": "YOUR_API_KEY_HERE",
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }
        
        # Note: Since we don't have a real API key, this will fail
        # and we'll fall back to synthetic data
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code == 200:
            odds_data = response.json()
            
            # Process the response into a DataFrame
            # Implementation depends on the API response format
            odds_list = []
            
            for game in odds_data:
                game_date = game['commence_time'].split('T')[0]
                
                # Extract teams
                home_team = None
                away_team = None
                
                for team in game['home_team']:
                    if 'home_team' in team:
                        home_team = team['name']
                    else:
                        away_team = team['name']
                
                # Extract odds from bookmakers
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker['key'] == 'fanduel':  # Use FanDuel as example
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                # Moneyline odds
                                home_odds = None
                                away_odds = None
                                
                                for outcome in market.get('outcomes', []):
                                    if outcome['name'] == home_team:
                                        home_odds = outcome['price']
                                    else:
                                        away_odds = outcome['price']
                            
                            elif market['key'] == 'spreads':
                                # Spread odds
                                spread = None
                                
                                for outcome in market.get('outcomes', []):
                                    if outcome['name'] == home_team:
                                        spread = outcome['point']
                
                odds_list.append({
                    'GAME_DATE': game_date,
                    'HOME_TEAM': home_team,
                    'AWAY_TEAM': away_team,
                    'HOME_ODDS': home_odds,
                    'AWAY_ODDS': away_odds,
                    'SPREAD': spread
                })
            
            odds_df = pd.DataFrame(odds_list)
            
            # Save to cache file
            os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
            odds_df.to_csv(cache_file, index=False)
            print(f"Saved betting odds to {cache_file}")
            
            return odds_df
        
    except Exception as e:
        print(f"Error fetching betting odds: {e}")
        print("Falling back to synthetic odds data...")
    
    # If we can't get real data, generate synthetic odds
    odds_df = generate_synthetic_odds(season)
    
    # Save to cache file
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    odds_df.to_csv(cache_file, index=False)
    print(f"Saved synthetic betting odds to {cache_file}")
    
    return odds_df

def generate_synthetic_odds(season='2023-24', team_strength=None):
    """
    Generate synthetic betting odds based on real game data
    
    Parameters:
    -----------
    season : str
        NBA season in format 'YYYY-YY'
    team_strength : dict, optional
        Dictionary mapping team names to relative strength
        
    Returns:
    --------
    odds_df : pandas DataFrame
        DataFrame with synthetic betting odds
    """
    # Load real game data
    try:
        games_df = load_nba_data(season)
    except Exception as e:
        print(f"Error loading NBA data: {e}")
        # If we can't load real data, fall back to fully synthetic data
        return generate_synthetic_data()[1]
    
    # Extract home games
    home_games = games_df[games_df['MATCHUP'].str.contains('vs.', na=False)].copy()
    
    # Create team strength ratings if not provided
    if team_strength is None:
        # Calculate win percentage for each team
        team_records = games_df.groupby('TEAM_NAME').agg({
            'WL': lambda x: (x == 'W').mean()
        }).reset_index()
        
        team_strength = dict(zip(team_records['TEAM_NAME'], team_records['WL']))
    
    # Generate synthetic odds
    odds_data = []
    
    for _, game in home_games.iterrows():
        home_team = game['TEAM_NAME']
        
        # Extract away team from matchup
        matchup = game['MATCHUP']
        if isinstance(matchup, str) and 'vs.' in matchup:
            away_team = matchup.split('vs.')[1].strip()
        else:
            # Skip if matchup format is unexpected
            continue
        
        game_date = game['GAME_DATE']
        
        # Determine if home team won
        home_win = game['WL'] == 'W'
        
        # Generate realistic odds based on team strength
        home_strength = team_strength.get(home_team, 0.5)
        away_strength = team_strength.get(away_team, 0.5)
        
        # Calculate odds (stronger teams get lower/negative odds)
        strength_diff = home_strength - away_strength
        
        if strength_diff > 0.1:  # Home team is favored
            home_odds = np.random.uniform(-220, -110)
            away_odds = np.random.uniform(110, 240)
            spread = np.random.uniform(-7.5, -1.5)
        elif strength_diff < -0.1:  # Away team is favored
            home_odds = np.random.uniform(110, 240)
            away_odds = np.random.uniform(-220, -110)
            spread = np.random.uniform(1.5, 7.5)
        else:  # Even matchup
            if np.random.random() < 0.5:
                home_odds = np.random.uniform(-130, -100)
                away_odds = np.random.uniform(100, 130)
                spread = np.random.uniform(-3, -1)
            else:
                home_odds = np.random.uniform(100, 130)
                away_odds = np.random.uniform(-130, -100)
                spread = np.random.uniform(1, 3)
        
        # Actual game result slightly influences odds (for more realism)
        if home_win and home_odds > 0:
            # Underdog home team won, make odds slightly less extreme
            home_odds = np.random.uniform(100, 160)
        elif not home_win and home_odds < 0:
            # Favored home team lost, make odds slightly less extreme
            home_odds = np.random.uniform(-180, -110)
        
        # Add over/under line
        over_under = np.random.uniform(215, 235)
            
        odds_data.append({
            'GAME_DATE': game_date,
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'HOME_ODDS': home_odds,
            'AWAY_ODDS': away_odds,
            'SPREAD': spread,
            'OVER_UNDER': over_under
        })
    
    # Convert to DataFrame
    odds_df = pd.DataFrame(odds_data)
    
    return odds_df

def generate_synthetic_data(n_teams=30, n_games=500):
    """
    Generate fully synthetic NBA game and odds data for testing
    
    Parameters:
    -----------
    n_teams : int
        Number of teams to generate
    n_games : int
        Number of games to generate
        
    Returns:
    --------
    games_df : pandas DataFrame
        DataFrame with synthetic NBA game data
    odds_df : pandas DataFrame
        DataFrame with synthetic betting odds data
    """
    print("Generating fully synthetic data for testing...")
    
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

def get_current_season_data():
    """
    Get data for the current NBA season
    
    Returns:
    --------
    games_df : pandas DataFrame
        DataFrame with current season NBA game data
    odds_df : pandas DataFrame
        DataFrame with current season betting odds data
    """
    # Determine current season
    current_date = datetime.now()
    
    if current_date.month >= 10:  # NBA season typically starts in October
        season_start = current_date.year
    else:
        season_start = current_date.year - 1
    
    season = f"{season_start}-{str(season_start + 1)[-2:]}"
    
    print(f"Getting data for current season: {season}")
    
    # Load NBA data
    games_df = load_nba_data(season)
    
    # Load betting odds
    odds_df = load_odds_data(season)
    
    return games_df, odds_df