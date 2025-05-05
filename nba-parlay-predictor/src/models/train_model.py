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
    Train and evaluate a logistic regression model with feature selection
    
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
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a pipeline with preprocessing, feature selection, and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42), 
            threshold='median'  # Use median feature importance as threshold
        )),
        ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=42))  # Add regularization
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Print selected features
    try:
        feature_selector = pipeline.named_steps['feature_selection']
        selected_features_mask = feature_selector.get_support()
        selected_features = X_train.columns[selected_features_mask]
        print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features:")
        for feature in selected_features:
            print(f"  - {feature}")
    except Exception as e:
        print(f"Could not print selected features: {e}")
    
    return pipeline