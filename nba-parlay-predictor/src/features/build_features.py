"""
Functions for feature engineering
"""
import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Create features for model training
    
    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed game data
        
    Returns:
    --------
    feature_df : pandas DataFrame
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    feature_df = df.copy()
    
    # 1. Basic statistical differences
    print("Creating basic statistical features...")
    feature_df['PTS_DIFF'] = feature_df['HOME_PTS'] - feature_df['AWAY_PTS']
    feature_df['FG_PCT_DIFF'] = feature_df['HOME_FG_PCT'] - feature_df['AWAY_FG_PCT']
    feature_df['FT_PCT_DIFF'] = feature_df['HOME_FT_PCT'] - feature_df['AWAY_FT_PCT']
    feature_df['REB_DIFF'] = feature_df['HOME_REB'] - feature_df['AWAY_REB']
    feature_df['AST_DIFF'] = feature_df['HOME_AST'] - feature_df['AWAY_AST']
    feature_df['STL_DIFF'] = feature_df['HOME_STL'] - feature_df['AWAY_STL']
    feature_df['BLK_DIFF'] = feature_df['HOME_BLK'] - feature_df['AWAY_BLK']
    feature_df['TOV_DIFF'] = feature_df['HOME_TOV'] - feature_df['AWAY_TOV']
    
    # 2. Create efficiency metrics
    print("Creating efficiency metrics...")
    # Offensive efficiency (points per possession approximation)
    feature_df['HOME_OFF_EFF'] = feature_df['HOME_PTS'] / (feature_df['HOME_FG_PCT'] * feature_df['HOME_TOV'])
    feature_df['AWAY_OFF_EFF'] = feature_df['AWAY_PTS'] / (feature_df['AWAY_FG_PCT'] * feature_df['AWAY_TOV'])
    feature_df['OFF_EFF_DIFF'] = feature_df['HOME_OFF_EFF'] - feature_df['AWAY_OFF_EFF']
    
    # Handle potential inf/NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.mean())
    
    # 3. Add betting odds features if available
    if 'SPREAD' in feature_df.columns:
        print("Adding betting odds features...")
        # Convert American odds to implied probability
        feature_df['HOME_IMPLIED_PROB'] = feature_df.apply(
            lambda x: (100 / abs(x['HOME_ODDS'])) if x['HOME_ODDS'] > 0 
            else (abs(x['HOME_ODDS']) / (abs(x['HOME_ODDS']) + 100)), axis=1
        )
        feature_df['AWAY_IMPLIED_PROB'] = feature_df.apply(
            lambda x: (100 / abs(x['AWAY_ODDS'])) if x['AWAY_ODDS'] > 0 
            else (abs(x['AWAY_ODDS']) / (abs(x['AWAY_ODDS']) + 100)), axis=1
        )
        
        # Calculate the over-round (bookmaker's margin)
        feature_df['OVERROUND'] = feature_df['HOME_IMPLIED_PROB'] + feature_df['AWAY_IMPLIED_PROB']
    
    # 4. For a more advanced implementation (not for the one-day project), 
    # you would calculate moving averages and team form here
    
    print(f"Feature engineering complete. Total features: {len(feature_df.columns)}")
    return feature_df