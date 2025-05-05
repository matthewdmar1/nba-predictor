import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.train_model import train_model
from src.models.evaluate import generate_parlays, simulate_roi

def time_based_validation(data, target_column, feature_columns, time_column, 
                         min_train_size=0.5, n_splits=5):
    """
    Perform time-based cross-validation
    
    Parameters:
    -----------
    data : pandas DataFrame
        Full dataset with timestamp column
    target_column : str
        Name of the target column
    feature_columns : list
        List of feature column names
    time_column : str
        Name of the timestamp column
    min_train_size : float
        Minimum proportion of data used for initial training
    n_splits : int
        Number of validation splits
    
    Returns:
    --------
    results : dict
        Dictionary with validation results
    """
    # Sort data by time
    sorted_data = data.sort_values(time_column).reset_index(drop=True)
    
    # Prepare results containers
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roi': [],
        'train_size': [],
        'test_size': [],
        'test_start_date': [],
        'test_end_date': []
    }
    
    # Calculate split points
    n_samples = len(sorted_data)
    initial_train_size = int(n_samples * min_train_size)
    remaining_samples = n_samples - initial_train_size
    test_size = remaining_samples // n_splits
    
    # Perform time-based validation
    for i in range(n_splits):
        # Define split points
        train_end = initial_train_size + i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        
        # Get train and test sets
        train_data = sorted_data.iloc[:train_end]
        test_data = sorted_data.iloc[test_start:test_end]
        
        # Extract features and target
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Store metadata about split
        test_start_date = test_data[time_column].min()
        test_end_date = test_data[time_column].max()
        metrics['train_size'].append(len(train_data))
        metrics['test_size'].append(len(test_data))
        metrics['test_start_date'].append(test_start_date)
        metrics['test_end_date'].append(test_end_date)
        
        # Train and evaluate model
        model = train_model(X_train, y_train, cv_folds=3)
        y_pred = model.predict(X_test)
        
        # Calculate standard metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        
        # Generate parlays for this time period
        teams_info_test = test_data[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
        probabilities = model.predict_proba(X_test)[:, 1]
        
        results_df = teams_info_test.copy()
        results_df['predicted_home_win'] = y_pred
        results_df['home_win_probability'] = probabilities
        results_df['correct_prediction'] = (results_df['HOME_WIN'] == results_df['predicted_home_win'])
        
        parlays = generate_parlays(results_df)
        
        # Simulate ROI for this time period
        if parlays:
            roi_results = simulate_roi(parlays, results_df)
            metrics['roi'].append(roi_results['total_roi'])
        else:
            metrics['roi'].append(0.0)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(metrics)
    
    # Calculate overall average metrics
    overall = {
        'avg_accuracy': results_df['accuracy'].mean(),
        'avg_precision': results_df['precision'].mean(),
        'avg_recall': results_df['recall'].mean(),
        'avg_f1': results_df['f1'].mean(),
        'avg_roi': results_df['roi'].mean(),
        'results_by_period': results_df
    }
    
    return overall

def plot_temporal_validation_results(results):
    """Plot temporal validation results"""
    results_df = results['results_by_period']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy, precision, recall, f1
    metrics_df = results_df.melt(
        id_vars=['test_start_date', 'test_end_date'],
        value_vars=['accuracy', 'precision', 'recall', 'f1'],
        var_name='Metric', value_name='Value'
    )
    
    sns.lineplot(
        data=metrics_df, 
        x='test_start_date', 
        y='Value', 
        hue='Metric',
        marker='o',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Classification Metrics Over Time')
    axes[0, 0].set_xlabel('Test Period Start')
    axes[0, 0].set_ylabel('Metric Value')
    
    # Plot ROI over time
    sns.lineplot(
        data=results_df,
        x='test_start_date',
        y='roi',
        marker='o',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('ROI Over Time')
    axes[0, 1].set_xlabel('Test Period Start')
    axes[0, 1].set_ylabel('ROI (%)')
    axes[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Plot train vs test size
    size_df = results_df.melt(
        id_vars=['test_start_date'],
        value_vars=['train_size', 'test_size'],
        var_name='Dataset', value_name='Size'
    )
    
    sns.barplot(
        data=size_df,
        x='test_start_date',
        y='Size',
        hue='Dataset',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Dataset Sizes')
    axes[1, 0].set_xlabel('Test Period Start')
    axes[1, 0].set_ylabel('Number of Samples')
    
    # Plot average metrics
    avg_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROI (%)'],
        'Value': [
            results['avg_accuracy'], 
            results['avg_precision'],
            results['avg_recall'],
            results['avg_f1'],
            results['avg_roi']
        ]
    })
    
    sns.barplot(
        data=avg_metrics,
        x='Metric',
        y='Value',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Average Metrics')
    axes[1, 1].set_xlabel('Metric')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def test_parlay_strategy(data, target_column, feature_columns, time_column,
                       min_confidence=0.65, max_games=3, stake=100):
    """
    Test parlay betting strategy with different parameters
    
    Parameters:
    -----------
    data : pandas DataFrame
        Full dataset with timestamp column
    target_column : str
        Name of the target column
    feature_columns : list
        List of feature column names
    time_column : str
        Name of the timestamp column
    min_confidence : float
        Minimum confidence threshold for including games in parlays
    max_games : int
        Maximum number of games per parlay
    stake : float
        Stake amount per parlay
    
    Returns:
    --------
    results : dict
        Dictionary with strategy test results
    """
    # Sort data by time
    sorted_data = data.sort_values(time_column).reset_index(drop=True)
    
    # Use first 70% for training
    train_size = int(len(sorted_data) * 0.7)
    train_data = sorted_data.iloc[:train_size]
    test_data = sorted_data.iloc[train_size:]
    
    # Extract features and target
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Create results dataframe
    teams_info_test = test_data[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    results_df = teams_info_test.copy()
    results_df['predicted_home_win'] = y_pred
    results_df['home_win_probability'] = probabilities
    results_df['correct_prediction'] = (results_df['HOME_WIN'] == results_df['predicted_home_win'])
    
    # Test different confidence thresholds
    confidence_thresholds = np.arange(0.6, 0.95, 0.05)
    
    # Results container
    strategy_results = {
        'threshold': [],
        'num_parlays': [],
        'avg_parlay_size': [],
        'win_rate': [],
        'roi': [],
        'profit': []
    }
    
    for threshold in confidence_thresholds:
        parlays = generate_parlays(
            results_df, 
            min_prob=threshold, 
            max_games=max_games
        )
        
        if parlays:
            roi_results = simulate_roi(parlays, results_df, stake=stake)
            
            strategy_results['threshold'].append(threshold)
            strategy_results['num_parlays'].append(len(parlays))
            strategy_results['avg_parlay_size'].append(
                sum(p['parlay_size'] for p in parlays) / len(parlays)
            )
            win_rate = roi_results['roi_df']['wins'].sum() / len(roi_results['roi_df'])
            strategy_results['win_rate'].append(win_rate)
            strategy_results['roi'].append(roi_results['total_roi'])
            strategy_results['profit'].append(roi_results['total_profit'])
    
    return pd.DataFrame(strategy_results)