"""
Functions for evaluating the prediction model and generating parlays
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test, teams_info):
    """
    Evaluate model performance on test data
    
    Parameters:
    -----------
    model : scikit-learn model
        Trained classification model
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        True labels
    teams_info : pandas DataFrame
        Team information for reference
        
    Returns:
    --------
    results_df : pandas DataFrame
        Dataframe with prediction results
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (HOME_WIN)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Create results dataframe
    results_df = teams_info.copy()
    results_df['predicted_home_win'] = y_pred
    results_df['home_win_probability'] = probabilities
    results_df['correct_prediction'] = (results_df['HOME_WIN'] == results_df['predicted_home_win'])
    
    return results_df

def generate_parlays(results_df, min_prob=0.65, max_games=3, min_games=2):
    """
    Generate parlay recommendations based on model predictions
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Dataframe with prediction results
    min_prob : float
        Minimum probability threshold for including a game in a parlay
    max_games : int
        Maximum number of games in a parlay
    min_games : int
        Minimum number of games in a parlay
        
    Returns:
    --------
    parlays : list of dict
        List of parlay recommendations
    """
    # Filter games with high confidence predictions
    high_confidence = results_df[
        ((results_df['predicted_home_win'] == True) & (results_df['home_win_probability'] >= min_prob)) | 
        ((results_df['predicted_home_win'] == False) & (results_df['home_win_probability'] <= (1 - min_prob)))
    ].copy()
    
    # Calculate the adjusted probability for away wins
    high_confidence['adjusted_probability'] = high_confidence.apply(
        lambda x: x['home_win_probability'] if x['predicted_home_win'] else (1 - x['home_win_probability']), 
        axis=1
    )
    
    # Sort by adjusted probability (highest confidence first)
    high_confidence = high_confidence.sort_values('adjusted_probability', ascending=False)
    
    # Check if we have enough high-confidence games
    if len(high_confidence) < min_games:
        print(f"Not enough high-confidence games found. Only {len(high_confidence)} games meet the minimum probability threshold of {min_prob}.")
        return []
    
    # Generate parlays (for this simplified version, we'll create one optimal parlay)
    parlays = []
    
    # Strategy 1: Highest combined probability parlay
    for size in range(min(max_games, len(high_confidence)), min_games-1, -1):
        best_games = high_confidence.iloc[:size]
        
        # Calculate combined probability (multiply individual probabilities)
        combined_prob = best_games['adjusted_probability'].product()
        
        # Create parlay
        parlay = {
            'combined_probability': combined_prob,
            'parlay_size': size,
            'games': best_games.to_dict('records')
        }
        
        parlays.append(parlay)
    
    # Strategy 2: Alternative parlay with different games
    if len(high_confidence) > max_games + min_games:
        alt_games = high_confidence.iloc[max_games:max_games + min_games]
        combined_prob = alt_games['adjusted_probability'].product()
        
        alt_parlay = {
            'combined_probability': combined_prob,
            'parlay_size': len(alt_games),
            'games': alt_games.to_dict('records')
        }
        
        parlays.append(alt_parlay)
    
    print(f"Generated {len(parlays)} parlay recommendations.")
    return parlays

def simulate_roi(parlays, results_df, iterations=1000, stake=100):
    """
    Simulate Return on Investment (ROI) for parlay recommendations
    
    Parameters:
    -----------
    parlays : list of dict
        List of parlay recommendations
    results_df : pandas DataFrame
        Dataframe with actual game results
    iterations : int
        Number of simulation iterations
    stake : float
        Stake amount per parlay
        
    Returns:
    --------
    roi_results : dict
        Results of ROI simulation
    """
    print(f"Simulating ROI over {iterations} iterations with ${stake} stake per parlay...")
    
    # Function to calculate parlay odds (American format)
    def calculate_parlay_odds(probabilities):
        decimal_odds = [1 / p for p in probabilities]
        combined_decimal = np.prod(decimal_odds)
        american_odds = (combined_decimal - 1) * 100
        return american_odds
    
    # Simulate ROI for each parlay
    roi_data = []
    
    for parlay_idx, parlay in enumerate(parlays):
        wins = 0
        total_profit = 0
        game_probabilities = []
        
        # Extract game probabilities for odds calculation
        for game in parlay['games']:
            prob = game['home_win_probability'] if game['predicted_home_win'] else (1 - game['home_win_probability'])
            game_probabilities.append(prob)
        
        # Calculate parlay odds
        parlay_odds = calculate_parlay_odds(game_probabilities)
        
        # Calculate expected payout
        if parlay_odds >= 0:
            payout = stake + (stake * parlay_odds / 100)
        else:
            payout = stake + (stake * 100 / abs(parlay_odds))
        
        # Determine if parlay would have won (all predictions correct)
        all_correct = True
        for game in parlay['games']:
            game_id = game['GAME_DATE'].strftime('%Y-%m-%d') + '_' + game['HOME_TEAM'] + '_' + game['AWAY_TEAM']
            actual_result = results_df[
                (results_df['GAME_DATE'] == game['GAME_DATE']) & 
                (results_df['HOME_TEAM'] == game['HOME_TEAM']) & 
                (results_df['AWAY_TEAM'] == game['AWAY_TEAM'])
            ]['HOME_WIN'].values[0]
            
            # Check if prediction was correct
            if game['predicted_home_win'] != actual_result:
                all_correct = False
                break
        
        # Record result
        if all_correct:
            wins = 1
            profit = payout - stake
        else:
            profit = -stake
        
        # Store simulation results
        roi_data.append({
            'parlay_idx': parlay_idx + 1,
            'combined_probability': parlay['combined_probability'],
            'parlay_size': parlay['parlay_size'],
            'american_odds': parlay_odds,
            'expected_payout': payout,
            'wins': wins,
            'profit': profit,
            'roi': (profit / stake) * 100
        })
    
    # Create dataframe with results
    roi_df = pd.DataFrame(roi_data)
    
    # Calculate overall results
    total_investment = stake * len(parlays)
    total_profit = roi_df['profit'].sum()
    total_roi = (total_profit / total_investment) * 100
    
    print(f"Total parlays: {len(parlays)}")
    print(f"Winning parlays: {roi_df['wins'].sum()}")
    print(f"Total investment: ${total_investment:.2f}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Overall ROI: {total_roi:.2f}%")
    
    # Return results
    roi_results = {
        'roi_df': roi_df,
        'total_investment': total_investment,
        'total_profit': total_profit,
        'total_roi': total_roi
    }
    
    return roi_results