"""
Functions for training the prediction model
"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def train_model(X_train, y_train, cv_folds=5):
    """
    Train and evaluate a logistic regression model
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target variable
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    model : fitted scikit-learn Pipeline
        Trained model with preprocessing
    """
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return pipeline