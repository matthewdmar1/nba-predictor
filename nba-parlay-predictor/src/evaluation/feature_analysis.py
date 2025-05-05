import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

def analyze_features(data, target_column, feature_columns, model=None, top_n=15):
    """
    Analyze features and their relationships with the target
    
    Parameters:
    -----------
    data : pandas DataFrame
        Dataset with features and target
    target_column : str
        Name of the target column
    feature_columns : list
        List of feature column names
    model : trained model, optional
        Trained model for permutation importance
    top_n : int
        Number of top features to show
        
    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    # Extract features and target
    X = data[feature_columns]
    y = data[target_column]
    
    # Scale features for consistent analysis
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    # Calculate correlation with target
    correlation = {}
    for col in X.columns:
        correlation[col] = data[col].corr(data[target_column])
    
    # Calculate mutual information (non-linear relationships)
    mutual_info = mutual_info_classif(X, y, random_state=42)
    mi_scores = dict(zip(X.columns, mutual_info))
    
    # Calculate permutation importance if model provided
    perm_importance = None
    if model is not None:
        try:
            perm_imp = permutation_importance(
                model, X, y, n_repeats=10, random_state=42
            )
            perm_importance = dict(zip(X.columns, perm_imp.importances_mean))
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
    
    # Combine scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'correlation': [correlation[col] for col in X.columns],
        'mutual_info': [mi_scores[col] for col in X.columns]
    })
    
    if perm_importance is not None:
        feature_scores['permutation_importance'] = [
            perm_importance[col] for col in X.columns
        ]
    
    # Calculate feature correlations
    feature_correlation = X_scaled.corr()
    
    return {
        'feature_scores': feature_scores,
        'feature_correlation': feature_correlation
    }

def plot_feature_analysis(analysis_results, top_n=15):
    """
    Plot feature analysis results
    
    Parameters:
    -----------
    analysis_results : dict
        Results from analyze_features function
    top_n : int
        Number of top features to show
        
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    feature_scores = analysis_results['feature_scores']
    feature_correlation = analysis_results['feature_correlation']
    
    figs = []
    
    # Plot 1: Feature importance by different metrics
    has_perm_importance = 'permutation_importance' in feature_scores.columns
    
    fig1, axes1 = plt.subplots(
        1, 2 + int(has_perm_importance), 
        figsize=(15, 8)
    )
    
    # Sort by correlation
    correlation_df = feature_scores.sort_values('correlation', key=abs, ascending=False).head(top_n)
    sns.barplot(
        data=correlation_df,
        x='correlation',
        y='feature',
        ax=axes1[0]
    )
    axes1[0].set_title('Feature Correlation with Target')
    axes1[0].set_xlabel('Correlation')
    axes1[0].set_ylabel('Feature')
    
    # Sort by mutual information
    mi_df = feature_scores.sort_values('mutual_info', ascending=False).head(top_n)
    sns.barplot(
        data=mi_df,
        x='mutual_info',
        y='feature',
        ax=axes1[1]
    )
    axes1[1].set_title('Feature Mutual Information with Target')
    axes1[1].set_xlabel('Mutual Information')
    axes1[1].set_ylabel('Feature')
    
    # Sort by permutation importance if available
    if has_perm_importance:
        perm_df = feature_scores.sort_values('permutation_importance', ascending=False).head(top_n)
        sns.barplot(
            data=perm_df,
            x='permutation_importance',
            y='feature',
            ax=axes1[2]
        )
        axes1[2].set_title('Feature Permutation Importance')
        axes1[2].set_xlabel('Permutation Importance')
        axes1[2].set_ylabel('Feature')
    
    plt.tight_layout()
    figs.append(fig1)
    
    # Plot 2: Feature correlation heatmap
    # Get top features
    top_features = feature_scores.sort_values('mutual_info', ascending=False)['feature'].head(top_n).tolist()
    
    # Create correlation heatmap for top features
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        feature_correlation.loc[top_features, top_features],
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        ax=ax2
    )
    
    ax2.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    figs.append(fig2)
    
    return figs