import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.models.evaluate import evaluate_model, generate_parlays, simulate_roi

def compare_models(data, target_column, feature_columns, test_size=0.3, random_state=42):
    """
    Compare multiple model architectures on the same dataset
    
    Parameters:
    -----------
    data : pandas DataFrame
        Full dataset
    target_column : str
        Name of the target column
    feature_columns : list
        List of feature column names
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with comparison results
    """
    # Split data
    X = data[feature_columns]
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Keep team info for parlay generation
    teams_info = data[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]
    _, teams_info_test = train_test_split(
        teams_info, test_size=test_size, random_state=random_state
    )
    
    # Define models to compare
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=random_state))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'))
        ])
    }
    
    # Prepare results containers
    model_metrics = []
    parlay_metrics = []
    roc_curves = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Store metrics
        model_metrics.append({
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr}
        
        # Create results dataframe for parlay evaluation
        results_df = teams_info_test.copy()
        results_df['predicted_home_win'] = y_pred
        results_df['home_win_probability'] = y_prob
        results_df['correct_prediction'] = (results_df['HOME_WIN'] == results_df['predicted_home_win'])
        
        # Generate parlays
        parlays = generate_parlays(results_df)
        
        # Skip if no parlays were generated
        if not parlays:
            continue
            
        # Simulate ROI
        roi_results = simulate_roi(parlays, results_df)
        
        # Store parlay metrics
        parlay_metrics.append({
            'model': model_name,
            'num_parlays': len(parlays),
            'avg_parlay_size': sum(p['parlay_size'] for p in parlays) / len(parlays),
            'avg_probability': sum(p['combined_probability'] for p in parlays) / len(parlays),
            'win_rate': roi_results['roi_df']['wins'].sum() / len(roi_results['roi_df']),
            'total_profit': roi_results['total_profit'],
            'roi': roi_results['total_roi']
        })
    
    # Convert to DataFrames
    model_metrics_df = pd.DataFrame(model_metrics)
    parlay_metrics_df = pd.DataFrame(parlay_metrics)
    
    return {
        'model_metrics': model_metrics_df,
        'parlay_metrics': parlay_metrics_df,
        'roc_curves': roc_curves,
        'models': models
    }

def plot_model_comparison(comparison_results):
    """
    Plot model comparison results
    
    Parameters:
    -----------
    comparison_results : dict
        Results from compare_models function
        
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    model_metrics = comparison_results['model_metrics']
    parlay_metrics = comparison_results['parlay_metrics']
    roc_curves = comparison_results['roc_curves']
    
    figs = []
    
    # Plot 1: Model metrics comparison
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    melted_metrics = model_metrics.melt(
        id_vars=['model'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    
    sns.barplot(
        data=melted_metrics,
        x='model',
        y='Value',
        hue='Metric',
        ax=ax1
    )
    
    ax1.set_title('Model Performance Metrics Comparison')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Metric Value')
    ax1.set_ylim(0, 1)
    ax1.legend(title='Metric')
    
    plt.tight_layout()
    figs.append(fig1)
    
    # Plot 2: ROC curves
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for model_name, curve_data in roc_curves.items():
        plt.plot(
            curve_data['fpr'],
            curve_data['tpr'],
            label=f"{model_name} (AUC = {model_metrics[model_metrics['model'] == model_name]['roc_auc'].values[0]:.3f})"
        )
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax2.set_title('ROC Curves')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    figs.append(fig2)
    
    # Plot 3: Parlay metrics
    if len(parlay_metrics) > 0:
        fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROI and profit
        financial_metrics = parlay_metrics[['model', 'total_profit', 'roi']].melt(
            id_vars=['model'],
            var_name='Metric',
            value_name='Value'
        )
        
        sns.barplot(
            data=financial_metrics,
            x='model',
            y='Value',
            hue='Metric',
            ax=axes3[0]
        )
        
        axes3[0].set_title('Parlay Financial Performance')
        axes3[0].set_xlabel('Model')
        axes3[0].set_ylabel('Value')
        axes3[0].tick_params(axis='x', rotation=45)
        
        # Win rate and probability
        probability_metrics = parlay_metrics[['model', 'win_rate', 'avg_probability']].melt(
            id_vars=['model'],
            var_name='Metric',
            value_name='Value'
        )
        
        sns.barplot(
            data=probability_metrics,
            x='model',
            y='Value',
            hue='Metric',
            ax=axes3[1]
        )
        
        axes3[1].set_title('Parlay Probability Performance')
        axes3[1].set_xlabel('Model')
        axes3[1].set_ylabel('Value')
        axes3[1].set_ylim(0, 1)
        axes3[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        figs.append(fig3)
    
    return figs