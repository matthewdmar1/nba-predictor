import joblib
import os
import numpy as np

print("=== NBA MODEL TESTING SCRIPT ===")

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Path to your model file - adjust as needed
model_path = "model.pkl"  # If in same directory
# Uncomment the line below if you need to use the relative path
# model_path = "../../nba-parlay-predictor/results/models/model.pkl"  

# Check if file exists
if os.path.exists(model_path):
    print(f"Model file found: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")
    
    try:
        # Try to load the model
        model = joblib.load(model_path)
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Print model details
        print("\nModel details:")
        print(model)
        
        # Check if the model has expected methods
        print("\nModel capabilities:")
        print(f"Has predict method: {'predict' in dir(model)}")
        print(f"Has predict_proba method: {'predict_proba' in dir(model)}")
        
        # Try to make a prediction with dummy data
        try:
            # Try different feature sizes to find the right one
            feature_sizes = [5, 10, 15, 20, 25, 30]
            
            for n_features in feature_sizes:
                print(f"\nTrying prediction with {n_features} features...")
                dummy_features = np.random.random(size=(1, n_features))
                
                # Try prediction
                try:
                    prediction = model.predict(dummy_features)
                    print(f"Prediction result: {prediction}")
                    print(f"SUCCESS! Your model expects {n_features} features.")
                    
                    # Try probability prediction
                    try:
                        probabilities = model.predict_proba(dummy_features)
                        print(f"Probability result: {probabilities}")
                        
                        # If we got here, we found the right feature size
                        break
                    except Exception as e:
                        print(f"Probability prediction error: {e}")
                        
                except Exception as e:
                    print(f"Prediction error with {n_features} features: {e}")
            
            print("\nIMPORTANT: If no feature size worked, you might need to:")
            print("1. Check if your model requires specific feature names")
            print("2. Verify that you saved the complete model pipeline")
            print("3. Retrain and resave your model")
            
        except Exception as e:
            print(f"Error during prediction test: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at: {model_path}")
    
    # Check parent directories
    parent_paths = [
        "../model.pkl",
        "../../model.pkl",
        "../nba-parlay-predictor/results/models/model.pkl",
        "../../nba-parlay-predictor/results/models/model.pkl"
    ]
    
    print("\nChecking other possible locations:")
    for path in parent_paths:
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path)
        else:
            size = "N/A"
        print(f"- {path}: Exists: {exists}, Size: {size}")

# Add this at the end of your script
if hasattr(model, 'feature_names_in_'):
    print("\nEXACT FEATURE NAMES expected by the model:")
    for i, name in enumerate(model.feature_names_in_):
        print(f"{i+1}. {name}")
elif hasattr(model, 'steps') and hasattr(model.steps[0][1], 'feature_names_in_'):
    print("\nEXACT FEATURE NAMES expected by the model:")
    for i, name in enumerate(model.steps[0][1].feature_names_in_):
        print(f"{i+1}. {name}")
else:
    print("\nModel doesn't have explicit feature names")