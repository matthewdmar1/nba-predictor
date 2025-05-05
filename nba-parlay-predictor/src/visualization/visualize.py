"""
Functions for visualizing model results and betting statistics
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=15, figsize=(12, 8), save_path=None):
    """
    Plot feature importances for the trained model
    
    Parameters:
    -----------
    model : scikit-learn model
        Trained classification model (must have .feature_importances_ or .coef_ attribute)
    feature_names : array-like
        Names of features
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Extract feature importances
    try:
        # For pipeline with feature_selection, extract selected feature importances
        if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:
            feature_selector = model.named_steps['feature_selection']
            selected_features_mask = feature_selector.get_support()
            selected_feature_names = feature_names[selected_features_mask]
            
            if hasattr(model.named_steps['classifier'], 'coef_'):
                importances = np.abs(model.named_steps['classifier'].coef_[0])
                feature_names = selected_feature_names
            else:
                raise AttributeError("Classifier does not have interpretable feature importances")
        # For tree-based models (Random Forest, XGBoost)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # For pipeline with classifier
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        # For linear models (Logistic Regression)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'coef_'):
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            # Create random importances just to demonstrate the plot
            print("Model does not have interpretable feature importances. Using random values for demonstration.")
            importances = np.random.random(size=len(feature_names))
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        # Create random importances as fallback
        importances = np.random.random(size=len(feature_names))
    
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Plot horizontal bar chart
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Calculate confusion matrix
    cm = sk_confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")

def plot_roi(roi_results, figsize=(10, 6), save_path=None):
    """
    Plot ROI simulation results
    
    Parameters:
    -----------
    roi_results : dict
        Results from simulate_roi function
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    roi_df = roi_results['roi_df']
    
    # Create figure with two subplots
    plt.figure(figsize=figsize)
    
    # Plot 1: Parlay Profit by Size (left side)
    plt.subplot(1, 2, 1)
    sizes = roi_df['parlay_size'].unique()
    profits = [roi_df[roi_df['parlay_size'] == size]['profit'].mean() for size in sizes]
    
    plt.bar(sizes, profits, color='skyblue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Average Profit by Parlay Size')
    plt.xlabel('Number of Games in Parlay')
    plt.ylabel('Profit ($)')
    
    # Plot 2: Win/Loss Pie (right side)
    plt.subplot(1, 2, 2)
    labels = ['Win', 'Loss']
    wins = roi_df['wins'].sum()
    losses = len(roi_df) - wins
    sizes = [wins, losses]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0) if wins > 0 else (0, 0.1)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f'Win/Loss Rate\nROI: {roi_results["total_roi"]:.2f}%')
    
    # Add text annotation in a separate text box
    text_str = (f"Investment: ${roi_results['total_investment']:.2f}\n"
                f"Profit: ${roi_results['total_profit']:.2f}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1.1, -0.1, text_str, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout(pad=3.0)
    
    # Save figure with explicit dimensions to avoid the size error
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', 
                    format='png', facecolor='white', 
                    transparent=False, pad_inches=0.5)
        
        print(f"ROI simulation plot saved to {save_path}")

def plot_win_probability_distribution(results_df, figsize=(12, 6), save_path=None):
    """
    Plot distribution of win probabilities for correct vs. incorrect predictions
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Dataframe with prediction results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Separate correct and incorrect predictions
    correct = results_df[results_df['correct_prediction']]
    incorrect = results_df[~results_df['correct_prediction']]
    
    # Plot probability distributions
    sns.histplot(correct['home_win_probability'], color='green', alpha=0.5, 
                 bins=20, label='Correct Predictions')
    sns.histplot(incorrect['home_win_probability'], color='red', alpha=0.5, 
                bins=20, label='Incorrect Predictions')
    
    plt.title('Win Probability Distribution: Correct vs. Incorrect Predictions')
    plt.xlabel('Predicted Home Win Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Win probability distribution plot saved to {save_path}")
    
    plt.show()

def plot_calibration_curve(y_true, y_prob, bins=10, figsize=(10, 8), save_path=None):
    """
    Plot calibration curve for predicted probabilities
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    bins : int
        Number of bins for probability
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Create dataframe with true labels and predicted probabilities
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    
    # Create bins
    df['bin'] = pd.cut(df['prob'], bins=bins)
    
    # Calculate fraction of positives in each bin
    cal_df = df.groupby('bin')['true'].agg(['mean', 'count']).reset_index()
    cal_df.columns = ['bin', 'fraction_of_positives', 'count']
    
    # Get bin centers
    cal_df['bin_center'] = cal_df['bin'].apply(lambda x: (x.left + x.right) / 2)
    
    # Plot calibration curve
    plt.scatter(cal_df['bin_center'], cal_df['fraction_of_positives'], 
               s=cal_df['count'] * 3, alpha=0.7)
    
    # Add line connecting points
    plt.plot(cal_df['bin_center'], cal_df['fraction_of_positives'], 'b-', alpha=0.5)
    
    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Fraction of Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve plot saved to {save_path}")
    
    plt.show()