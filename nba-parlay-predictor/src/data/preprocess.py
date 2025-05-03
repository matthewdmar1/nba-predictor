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
    print(f"Sample games date: {games_df['GAME_DATE'].iloc[0] if 'GAME_DATE' in games_df.columns and len(games_df) > 0 else 'No games data'}")
    print(f"Sample odds date: {odds_df['GAME_DATE'].iloc[0] if 'GAME_DATE' in odds_df.columns and len(odds_df) > 0 else 'No odds data'}")
    
    # Check if the dataframes have the required columns
    if len(games_df) == 0:
        print("Error: Empty games dataframe")
        return pd.DataFrame()
        
    # Make sure dates are in compatible formats
    if 'GAME_DATE' in games_df.columns and 'GAME_DATE' in odds_df.columns:
        if isinstance(games_df['GAME_DATE'].iloc[0], str):
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        if isinstance(odds_df['GAME_DATE'].iloc[0], str):
            odds_df['GAME_DATE'] = pd.to_datetime(odds_df['GAME_DATE'])
    
    # Extract home and away games
    # Try different match patterns - some datasets use "vs." while others might use "vs" without a period
    home_games = games_df[games_df['MATCHUP'].str.contains('vs', na=False)].copy()
    away_games = games_df[~games_df['MATCHUP'].str.contains('vs', na=False)].copy()
    
    print(f"Home games: {len(home_games)}, Away games: {len(away_games)}")
    
    # Print a few sample matchups to debug
    if len(home_games) > 0:
        print("Sample home matchups:")
        for i in range(min(3, len(home_games))):
            print(f"  {home_games['MATCHUP'].iloc[i]}")
    
    if len(away_games) > 0:
        print("Sample away matchups:")
        for i in range(min(3, len(away_games))):
            print(f"  {away_games['MATCHUP'].iloc[i]}")
    
    # Create a merged dataset of full games
    full_games = []
    
    # Iterate through home games and find corresponding away games
    for _, home_game in home_games.iterrows():
        try:
            home_team = home_game['TEAM_NAME']
            game_date = home_game['GAME_DATE']
            game_id = home_game['GAME_ID']
            
            # Try to find corresponding away game using game ID first
            away_game = away_games[away_games['GAME_ID'] == game_id]
            
            # If not found by game ID, try using date and matchup
            if len(away_game) == 0:
                # Extract opponent from matchup
                if 'vs.' in home_game['MATCHUP']:
                    opponent = home_game['MATCHUP'].split('vs.')[1].strip()
                elif 'vs' in home_game['MATCHUP']:
                    opponent = home_game['MATCHUP'].split('vs')[1].strip()
                else:
                    # Skip if can't parse opponent
                    continue
                
                # Look for away game with this opponent on same date
                away_game = away_games[
                    (away_games['GAME_DATE'] == game_date) & 
                    (away_games['TEAM_NAME'] == opponent)
                ]
            
            if len(away_game) >= 1:
                # Take the first matching away game
                away_game = away_game.iloc[0]
                away_team = away_game['TEAM_NAME']
                
                # Create a combined game record
                game_record = {
                    'GAME_DATE': game_date,
                    'GAME_ID': game_id,
                    'HOME_TEAM': home_team,
                    'AWAY_TEAM': away_team,
                    'HOME_PTS': home_game['PTS'],
                    'AWAY_PTS': away_game['PTS'],
                    'HOME_FG_PCT': home_game['FG_PCT'] if 'FG_PCT' in home_game else 0.0,
                    'AWAY_FG_PCT': away_game['FG_PCT'] if 'FG_PCT' in away_game else 0.0,
                    'HOME_FT_PCT': home_game['FT_PCT'] if 'FT_PCT' in home_game else 0.0,
                    'AWAY_FT_PCT': away_game['FT_PCT'] if 'FT_PCT' in away_game else 0.0,
                    'HOME_REB': home_game['REB'] if 'REB' in home_game else 0,
                    'AWAY_REB': away_game['REB'] if 'REB' in away_game else 0,
                    'HOME_AST': home_game['AST'] if 'AST' in home_game else 0,
                    'AWAY_AST': away_game['AST'] if 'AST' in away_game else 0,
                    'HOME_STL': home_game['STL'] if 'STL' in home_game else 0,
                    'AWAY_STL': away_game['STL'] if 'STL' in away_game else 0,
                    'HOME_BLK': home_game['BLK'] if 'BLK' in home_game else 0,
                    'AWAY_BLK': away_game['BLK'] if 'BLK' in away_game else 0,
                    'HOME_TOV': home_game['TOV'] if 'TOV' in home_game else 0,
                    'AWAY_TOV': away_game['TOV'] if 'TOV' in away_game else 0,
                    'HOME_PLUS_MINUS': home_game['PLUS_MINUS'] if 'PLUS_MINUS' in home_game else 0,
                    'AWAY_PLUS_MINUS': away_game['PLUS_MINUS'] if 'PLUS_MINUS' in away_game else 0,
                    'HOME_WIN': 1 if home_game['PTS'] > away_game['PTS'] else 0
                }
                
                full_games.append(game_record)
                
        except Exception as e:
            print(f"Error processing game: {e}")
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(full_games) if full_games else pd.DataFrame()
    
    print(f"Processed {len(processed_df)} complete games")
    
    # Merge with betting odds if available
    if not processed_df.empty and odds_df is not None and not odds_df.empty:
        try:
            # Ensure date formats are compatible for merging
            if 'GAME_DATE' in processed_df.columns and 'GAME_DATE' in odds_df.columns:
                if isinstance(processed_df['GAME_DATE'].iloc[0], str):
                    processed_df['GAME_DATE'] = pd.to_datetime(processed_df['GAME_DATE'])
                if isinstance(odds_df['GAME_DATE'].iloc[0], str):
                    odds_df['GAME_DATE'] = pd.to_datetime(odds_df['GAME_DATE'])
                
                # Print some debugging info about the merge keys
                print("Processed data sample GAME_DATE:")
                print(processed_df['GAME_DATE'].head())
                print("Odds data sample GAME_DATE:")
                print(odds_df['GAME_DATE'].head())
                
                print("Processed data HOME_TEAM values:")
                print(processed_df['HOME_TEAM'].unique()[:5])
                print("Odds data HOME_TEAM values:")
                print(odds_df['HOME_TEAM'].unique()[:5])
                
                # Try to normalize team names if needed
                # This is a simple approach - might need more complex matching in real data
                # ... (add team name normalization code if needed)
                
                # Merge datasets
                merged_df = processed_df.merge(
                    odds_df, 
                    on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
                    how='left'
                )
                
                print(f"Successfully merged odds data. Total games with odds: {merged_df['HOME_ODDS'].notna().sum()}")
                return merged_df
            else:
                print("Warning: Could not merge odds data due to missing GAME_DATE column")
                return processed_df
        except Exception as e:
            print(f"Warning: Could not merge odds data: {e}")
            print(f"Full error details: {e}")
            return processed_df
    else:
        print("No processed data or odds data to merge")
        # If no data was processed, return an empty DataFrame with the expected columns
        if processed_df.empty:
            columns = ['GAME_DATE', 'GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 
                      'HOME_FG_PCT', 'AWAY_FG_PCT', 'HOME_FT_PCT', 'AWAY_FT_PCT',
                      'HOME_REB', 'AWAY_REB', 'HOME_AST', 'AWAY_AST', 'HOME_STL', 
                      'AWAY_STL', 'HOME_BLK', 'AWAY_BLK', 'HOME_TOV', 'AWAY_TOV',
                      'HOME_PLUS_MINUS', 'AWAY_PLUS_MINUS', 'HOME_WIN']
            return pd.DataFrame(columns=columns)
        return processed_df