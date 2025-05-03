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
        # For tree-based models (Random Forest, XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # For pipeline with classifier
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        # For linear models (Logistic Regression)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'coef_'):
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            raise AttributeError("Model does not have interpretable feature importances")
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        return
    
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
    
    plt.show()

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

def plot_roi(roi_results, figsize=(12, 6), save_path=None):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Parlay Profit by Size
    sns.barplot(x='parlay_size', y='profit', data=roi_df, ax=ax1, palette='viridis')
    ax1.set_title('Profit by Parlay Size')
    ax1.set_xlabel('Number of Games in Parlay')
    ax1.set_ylabel('Profit ($)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    for i, row in roi_df.iterrows():
        color = 'green' if row['profit'] > 0 else 'red'
        ax1.text(i, row['profit'] + np.sign(row['profit']) * 5, 
                f"${row['profit']:.2f}", 
                ha='center', va='center', color=color, fontweight='bold')
    
    # Plot 2: Overall ROI
    labels = ['Win', 'Loss']
    wins = roi_df['wins'].sum()
    losses = len(roi_df) - wins
    sizes = [wins, losses]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0) if wins > 0 else (0, 0.1)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.axis('equal')
    ax2.set_title(f'Overall ROI: {roi_results["total_roi"]:.2f}%')
    
    # Add text with totals
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = (f"Total Investment: ${roi_results['total_investment']:.2f}\n"
                f"Total Profit: ${roi_results['total_profit']:.2f}")
    ax2.text(0, -1.2, text_str, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, ha='center')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROI simulation plot saved to {save_path}")
    
    plt.show()

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