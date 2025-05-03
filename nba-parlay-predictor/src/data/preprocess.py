"""
Functions for preprocessing NBA and betting data
"""
import pandas as pd
import numpy as np

def preprocess_data(games_df, odds_df):
    """
    Preprocess and merge NBA game data with betting odds
    
    Parameters:
    -----------
    games_df : pandas DataFrame
        NBA game data with team stats
    odds_df : pandas DataFrame
        Betting odds data
        
    Returns:
    --------
    processed_df : pandas DataFrame
        Processed and merged dataset
    """
    # Extract home and away games
    home_games = games_df[games_df['MATCHUP'].str.contains('vs.')].copy()
    away_games = games_df[games_df['MATCHUP'].str.contains('@')].copy()
    
    # Create a merged dataset of full games
    full_games = []
    
    # Iterate through home games and find corresponding away games
    for _, home_game in home_games.iterrows():
        home_team = home_game['TEAM_NAME']
        game_date = home_game['GAME_DATE']
        
        # Find the corresponding away game
        away_game = away_games[
            (away_games['GAME_DATE'] == game_date) & 
            (away_games['MATCHUP'].str.contains(f'@ {home_team}'))
        ]
        
        if len(away_game) == 1:
            away_game = away_game.iloc[0]
            away_team = away_game['TEAM_NAME']
            
            # Create a combined game record
            game_record = {
                'GAME_DATE': game_date,
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'HOME_PTS': home_game['PTS'],
                'AWAY_PTS': away_game['PTS'],
                'HOME_FG_PCT': home_game['FG_PCT'],
                'AWAY_FG_PCT': away_game['FG_PCT'],
                'HOME_FT_PCT': home_game['FT_PCT'],
                'AWAY_FT_PCT': away_game['FT_PCT'],
                'HOME_REB': home_game['REB'],
                'AWAY_REB': away_game['REB'],
                'HOME_AST': home_game['AST'],
                'AWAY_AST': away_game['AST'],
                'HOME_STL': home_game['STL'],
                'AWAY_STL': away_game['STL'],
                'HOME_BLK': home_game['BLK'],
                'AWAY_BLK': away_game['BLK'],
                'HOME_TOV': home_game['TOV'],
                'AWAY_TOV': away_game['TOV'],
                'HOME_PLUS_MINUS': home_game['PLUS_MINUS'],
                'AWAY_PLUS_MINUS': away_game['PLUS_MINUS'],
                'HOME_WIN': 1 if home_game['PTS'] > away_game['PTS'] else 0
            }
            
            full_games.append(game_record)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(full_games)
    
    # Merge with betting odds if available
    if odds_df is not None:
        try:
            # Convert dates to same format if necessary
            if isinstance(processed_df['GAME_DATE'].iloc[0], str):
                processed_df['GAME_DATE'] = pd.to_datetime(processed_df['GAME_DATE'])
            if isinstance(odds_df['GAME_DATE'].iloc[0], str):
                odds_df['GAME_DATE'] = pd.to_datetime(odds_df['GAME_DATE'])
                
            # Merge datasets
            processed_df = processed_df.merge(
                odds_df, 
                on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
                how='left'
            )
            
            print(f"Successfully merged odds data. Total games with odds: {processed_df['HOME_ODDS'].notna().sum()}")
        except Exception as e:
            print(f"Warning: Could not merge odds data: {e}")
    
    return processed_df