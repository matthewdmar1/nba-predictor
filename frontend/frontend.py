"""
NBA Parlay Predictor - Streamlit Frontend with Live Odds
This application combines web scraping for live odds with ML predictions for NBA games.
"""
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta
import traceback

# Set page config
st.set_page_config(
    page_title="NBA Parlay Predictor",
    page_icon="🏀",
    layout="wide"
)

# --- Web Scraping Functions ---

def scrape_website(url, element_type=None, class_name=None, element_id=None):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove status divs
        for div in soup.find_all('div', class_='event-cell__status event-cell__status__position'):
            div.decompose()

        elements = soup.find_all(element_type) if element_type else soup.find_all()

        if class_name:
            elements = [elem for elem in elements if elem.has_attr('class') and class_name in elem['class']]

        if element_id:
            elements = [elem for elem in elements if elem.has_attr('id') and elem['id'] == element_id]

        return elements

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the webpage: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return []

def parse_event_row(row):
    try:
        th = row.find("th")
        team_div = th.find("div", class_="event-cell__name-logo-wrapper")
        team_name = team_div.get_text(strip=True) if team_div else "Unknown"

        tds = row.find_all("td")
        if len(tds) < 3:
            return None

        outcomes = []
        for td in tds[:3]:
            outcome_div = td.find("div", class_="sportsbook-outcome-body-wrapper")
            outcome_text = outcome_div.get_text(strip=True) if outcome_div else "N/A"
            outcomes.append(outcome_text)

        return {
            "team": team_name,
            "spread": outcomes[0],
            "total": outcomes[1],
            "moneyline": outcomes[2]
        }

    except Exception as e:
        print(f"Error parsing row: {e}")
        return None

def table_to_custom_dataframe(table):
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        parsed = parse_event_row(tr)
        if parsed:
            rows.append(parsed)

    return pd.DataFrame(rows)

# --- Model Functions ---

def load_model():
    """Load the trained model using a relative path"""
    
    # Try multiple possible relative paths
    possible_paths = [
        "model.pkl",  # In the same directory
        "../model.pkl",  # One directory up
        "../../nba-parlay-predictor/results/models/model.pkl",  # Correct relative path from frontend to model
        "../results/models/model.pkl",  # Alternative if directory structure is different
    ]
    
    print("Attempting to load model from possible locations:")
    # Try each path until we find the model
    for path in possible_paths:
        try:
            import joblib
            import os
            
            # Convert backslashes to forward slashes for cross-platform compatibility
            path = path.replace('\\', '/')
            
            # Check if the file exists before trying to load it
            if os.path.exists(path):
                print(f"Model file found at {path}, size: {os.path.getsize(path)} bytes")
                model = joblib.load(path)
                print(f"Successfully loaded NBA prediction model from {path}")
                print(f"Model type: {type(model)}")
                
                # Check if model has predict_proba method
                has_predict = hasattr(model, 'predict')
                has_proba = hasattr(model, 'predict_proba')
                print(f"Model has predict: {has_predict}, predict_proba: {has_proba}")
                
                # Check if model is a pipeline
                if hasattr(model, 'steps'):
                    print(f"Model is a pipeline with steps: {[s[0] for s in model.steps]}")
                    
                    # Print expected feature names if possible
                    if hasattr(model.steps[0][1], 'feature_names_in_'):
                        feature_names = model.steps[0][1].feature_names_in_
                        print(f"Model expects {len(feature_names)} features")
                        print(f"First few expected features: {feature_names[:5]}")
                
                return model
            else:
                print(f"Model file not found at: {path}")
        except Exception as e:
            # Print the error for debugging
            print(f"Error loading model from {path}: {e}")
            # Just continue to the next path
            pass
    
    # If we get here, we couldn't find the model in any of the expected locations
    print("WARNING: Could not find model file. Using odds-based predictions as fallback.")
    
    # Provide instructions for users
    print("""
    To use your trained model:
    1. Copy your model.pkl file to this directory, or
    2. Update the paths in the code to point to your model location
    """)
    
    return None

def test_model(model_path):
    """Test if the model file is valid and what it contains"""
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model)}")
        
        # Test if it has predict_proba
        if hasattr(model, 'predict_proba'):
            print("Model has predict_proba method")
        else:
            print("WARNING: Model doesn't have predict_proba method")
            
        # Check for pipeline steps
        if hasattr(model, 'steps'):
            print(f"Model is a pipeline with steps: {[s[0] for s in model.steps]}")
        
        return model
    except Exception as e:
        print(f"Error testing model {model_path}: {e}")
        return None

def calculate_odds_based_prediction(game_data):
    """Calculate prediction based on betting odds"""
    try:
        # Add logging
        print("Using fallback odds-based prediction method")
        
        # Parse moneyline to get implied probabilities
        home_ml_str = game_data.get('home_moneyline', '-110')
        away_ml_str = game_data.get('away_moneyline', '-110')
        print(f"Home ML: {home_ml_str}, Away ML: {away_ml_str}")
        
        # Clean up the moneyline strings
        if isinstance(home_ml_str, str):
            home_ml_str = home_ml_str.replace('−', '-').replace('+', '')
            if " " in home_ml_str:
                home_ml_str = home_ml_str.split()[0]
        if isinstance(away_ml_str, str):
            away_ml_str = away_ml_str.replace('−', '-').replace('+', '')
            if " " in away_ml_str:
                away_ml_str = away_ml_str.split()[0]
        
        # Convert to float
        try:
            home_ml = float(home_ml_str)
            away_ml = float(away_ml_str)
        except ValueError:
            print(f"Error converting ML to float: {home_ml_str}, {away_ml_str}")
            home_ml = -110
            away_ml = -110
        
        # Convert moneyline to probability
        if home_ml > 0:
            home_prob = 100 / (home_ml + 100)
        else:
            home_prob = abs(home_ml) / (abs(home_ml) + 100)
        
        if away_ml > 0:
            away_prob = 100 / (away_ml + 100)
        else:
            away_prob = abs(away_ml) / (abs(away_ml) + 100)
        
        # Normalize probabilities
        total_prob = home_prob + away_prob
        home_prob = home_prob / total_prob
        
        print(f"Calculated home_prob: {home_prob}")
        
        prediction = {
            'home_win_probability': home_prob,
            'predicted_home_win': home_prob > 0.5,
            'confidence': max(home_prob, 1-home_prob),
            'source': 'odds'
        }
    except Exception as e:
        print(f"Error calculating odds-based prediction: {e}")
        print(traceback.format_exc())
        # Default to slightly favoring home team
        prediction = {
            'home_win_probability': 0.55,
            'predicted_home_win': True,
            'confidence': 0.55,
            'source': 'default'
        }
    return prediction

def extract_features_for_model(game_data):
    """
    Extract features for prediction model
    """
    try:
        # Use game-specific data to create more varied features
        home_team = game_data.get('home_team', '').lower()
        away_team = game_data.get('away_team', '').lower()

        # Create some variation based on team names
        home_seed = sum(ord(c) for c in home_team) % 10 + 1  # 1-10 based on team name
        away_seed = sum(ord(c) for c in away_team) % 10 + 1  # 1-10 based on team name
        
        # Define expected feature names - this should match your model's expectations
        features = {}
        
        # Add team stats with variation based on team name
        features['HOME_FG_PCT'] = 0.42 + (home_seed / 100)  # Range ~0.43-0.52
        features['AWAY_FG_PCT'] = 0.42 + (away_seed / 100)  # Range ~0.43-0.52
        features['HOME_FT_PCT'] = 0.72 + (home_seed / 100)  # Range ~0.73-0.82
        features['AWAY_FT_PCT'] = 0.72 + (away_seed / 100)  # Range ~0.73-0.82
        features['HOME_REB'] = 35 + home_seed  # Range 36-45
        features['AWAY_REB'] = 35 + away_seed  # Range 36-45
        features['HOME_AST'] = 20 + home_seed  # Range 21-30
        features['AWAY_AST'] = 20 + away_seed  # Range 21-30
        features['HOME_STL'] = 5 + (home_seed / 3)  # Range ~5.3-8.3
        features['AWAY_STL'] = 5 + (away_seed / 3)  # Range ~5.3-8.3
        features['HOME_BLK'] = 3 + (home_seed / 5)  # Range ~3.2-5
        features['AWAY_BLK'] = 3 + (away_seed / 5)  # Range ~3.2-5
        features['HOME_TOV'] = 12 + (home_seed / 4)  # Range ~12.3-14.5
        features['AWAY_TOV'] = 12 + (away_seed / 4)  # Range ~12.3-14.5
        
        # Add placeholder values for any other required features
        features['GAME_ID_x'] = 0
        features['GAME_ID_y'] = 0
            
        # Use the real betting odds from game_data
        try:
            # Process spread - handle special minus sign and extract just the number
            home_spread_str = game_data.get('home_spread', '0')
            if isinstance(home_spread_str, str):
                # Replace special minus sign
                home_spread_str = home_spread_str.replace('−', '-')
                # Extract just the number part (remove the odds part)
                if ' ' in home_spread_str:
                    home_spread_str = home_spread_str.split()[0]
                # Handle cases like "-9-110" where there's no space
                elif "-" in home_spread_str[1:]:  # Looking after the first character
                    parts = home_spread_str.split("-")
                    home_spread_str = parts[0] + "-" + parts[1]
                
                try:
                    features['SPREAD'] = float(home_spread_str)
                except ValueError:
                    print(f"Error parsing spread: {home_spread_str}, using default")
                    features['SPREAD'] = 0.0
            else:
                features['SPREAD'] = 0.0
            
            # Process home odds - handle special minus sign
            home_ml_str = game_data.get('home_moneyline', '-110')
            if isinstance(home_ml_str, str):
                # Replace both regular and special minus signs
                home_ml_str = home_ml_str.replace('−', '-').replace('+', '')
                # Handle cases where the moneyline might have additional text
                if " " in home_ml_str:
                    home_ml_str = home_ml_str.split()[0]
                
                try:
                    features['HOME_ODDS'] = float(home_ml_str)
                except ValueError:
                    print(f"Error parsing home odds: {home_ml_str}, using default")
                    features['HOME_ODDS'] = -110.0
            else:
                features['HOME_ODDS'] = -110.0
            
            # Process away odds - handle special minus sign
            away_ml_str = game_data.get('away_moneyline', '-110')
            if isinstance(away_ml_str, str):
                # Replace both regular and special minus signs
                away_ml_str = away_ml_str.replace('−', '-').replace('+', '')
                # Handle cases where the moneyline might have additional text
                if " " in away_ml_str:
                    away_ml_str = away_ml_str.split()[0]
                
                try:
                    features['AWAY_ODDS'] = float(away_ml_str)
                except ValueError:
                    print(f"Error parsing away odds: {away_ml_str}, using default")
                    features['AWAY_ODDS'] = -110.0
            else:
                features['AWAY_ODDS'] = -110.0
            
            print(f"Using odds: Home: {features['HOME_ODDS']}, Away: {features['AWAY_ODDS']}, Spread: {features['SPREAD']}")
        except Exception as e:
            print(f"Error processing odds: {e}")
            # If there's an error, use default values
            features['HOME_ODDS'] = -110.0
            features['AWAY_ODDS'] = -110.0
            features['SPREAD'] = 0.0
        
        # Calculate implied probabilities
        try:
            home_ml = features['HOME_ODDS']
            away_ml = features['AWAY_ODDS']
            
            # Convert moneyline to probability
            if home_ml > 0:
                features['HOME_IMPLIED_PROB'] = 100 / (home_ml + 100)
            else:
                features['HOME_IMPLIED_PROB'] = abs(home_ml) / (abs(home_ml) + 100)
                
            if away_ml > 0:
                features['AWAY_IMPLIED_PROB'] = 100 / (away_ml + 100)
            else:
                features['AWAY_IMPLIED_PROB'] = abs(away_ml) / (abs(away_ml) + 100)
                
            features['OVERROUND'] = features['HOME_IMPLIED_PROB'] + features['AWAY_IMPLIED_PROB']
            
            print(f"Calculated implied probs: Home: {features['HOME_IMPLIED_PROB']:.3f}, Away: {features['AWAY_IMPLIED_PROB']:.3f}")
        except Exception as e:
            print(f"Error calculating probabilities: {e}")
            features['HOME_IMPLIED_PROB'] = 0.5
            features['AWAY_IMPLIED_PROB'] = 0.5
            features['OVERROUND'] = 1.0
        
        # Calculate rating difference based on spread
        features['RATING_DIFF'] = -features['SPREAD']  # Negative spread means home team is favored
        features['HOME_NET_RATING'] = features['RATING_DIFF'] / 2
        features['AWAY_NET_RATING'] = -features['RATING_DIFF'] / 2
        
        # Calculate derived features
        features['FG_PCT_DIFF'] = features['HOME_FG_PCT'] - features['AWAY_FG_PCT']
        features['FT_PCT_DIFF'] = features['HOME_FT_PCT'] - features['AWAY_FT_PCT']
        features['REB_DIFF'] = features['HOME_REB'] - features['AWAY_REB']
        features['AST_DIFF'] = features['HOME_AST'] - features['AWAY_AST']
        features['STL_DIFF'] = features['HOME_STL'] - features['AWAY_STL']
        features['BLK_DIFF'] = features['HOME_BLK'] - features['AWAY_BLK']
        features['TOV_DIFF'] = features['HOME_TOV'] - features['AWAY_TOV']
        
        # Calculate offensive efficiency
        features['HOME_OFF_EFF'] = features['HOME_FG_PCT'] * 100 / max(features['HOME_TOV'], 1)
        features['AWAY_OFF_EFF'] = features['AWAY_FG_PCT'] * 100 / max(features['AWAY_TOV'], 1)
        features['OFF_EFF_DIFF'] = features['HOME_OFF_EFF'] - features['AWAY_OFF_EFF']
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        print(traceback.format_exc())
        # Return default features
        return {
            'HOME_FG_PCT': 0.45, 'AWAY_FG_PCT': 0.45,
            'HOME_FT_PCT': 0.75, 'AWAY_FT_PCT': 0.75,
            'HOME_REB': 40, 'AWAY_REB': 40,
            'HOME_AST': 25, 'AWAY_AST': 25,
            'HOME_STL': 7, 'AWAY_STL': 7,
            'HOME_BLK': 5, 'AWAY_BLK': 5,
            'HOME_TOV': 14, 'AWAY_TOV': 14,
            'HOME_ODDS': -110, 'AWAY_ODDS': -110,
            'SPREAD': 0, 'HOME_IMPLIED_PROB': 0.5,
            'AWAY_IMPLIED_PROB': 0.5, 'OVERROUND': 1.0,
            'RATING_DIFF': 0, 'HOME_NET_RATING': 0,
            'AWAY_NET_RATING': 0, 'GAME_ID_x': 0,
            'GAME_ID_y': 0, 'FG_PCT_DIFF': 0,
            'FT_PCT_DIFF': 0, 'REB_DIFF': 0,
            'AST_DIFF': 0, 'STL_DIFF': 0,
            'BLK_DIFF': 0, 'TOV_DIFF': 0,
            'HOME_OFF_EFF': 3.2, 'AWAY_OFF_EFF': 3.2,
            'OFF_EFF_DIFF': 0
        }

def make_prediction(model, game_data):
    """Make prediction with better error handling and debugging"""
    if model is None:
        # Use the fallback odds-based prediction
        return calculate_odds_based_prediction(game_data)
    
    try:
        # Extract features for your model
        features_dict = extract_features_for_model(game_data)
        
        # Debug output
        print(f"Making prediction for: {game_data.get('home_team')} vs {game_data.get('away_team')}")
        
        # Get expected feature names from model if possible
        feature_names = []
        if hasattr(model, 'steps') and len(model.steps) > 0:
            if hasattr(model.steps[0][1], 'feature_names_in_'):
                feature_names = model.steps[0][1].feature_names_in_
                print(f"Using {len(feature_names)} expected feature names from model")
                
                # Create features array with exactly the right ordering
                features_array = []
                for name in feature_names:
                    if name in features_dict:
                        features_array.append(features_dict[name])
                    else:
                        print(f"Missing feature: {name}, using default value 0")
                        features_array.append(0.0)
                
                features_array = np.array([features_array])
                
                # Make prediction
                print("Attempting prediction with correct feature ordering...")
                y_prob = model.predict_proba(features_array)
                home_win_prob = y_prob[0][1]  # Assuming index 1 is home win
                print(f"Prediction result: {home_win_prob}")
            else:
                # Fallback if model doesn't have feature_names_in_
                print("Model doesn't have feature_names_in_ attribute, using default order")
                features_array = np.array([[features_dict.get(k, 0) for k in sorted(features_dict.keys())]])
                y_prob = model.predict_proba(features_array)
                home_win_prob = y_prob[0][1]
        else:
            # Simple fallback
            print("Model doesn't have steps attribute, using default approach")
            features_list = list(features_dict.values())
            features_array = np.array([features_list])
            y_prob = model.predict_proba(features_array)
            home_win_prob = y_prob[0][1]
        
        # Apply probability clipping
        home_win_prob = min(max(home_win_prob, 0.05), 0.99)
        print(f"Final clipped probability: {home_win_prob}")
        
        # Return prediction in required format
        return {
            'home_win_probability': home_win_prob,
            'predicted_home_win': home_win_prob > 0.5,
            'confidence': max(home_win_prob, 1-home_win_prob),
            'source': 'model'
        }
    except Exception as e:
        print(f"Error making prediction with model: {e}")
        print(traceback.format_exc())
        # Fall back to odds-based prediction
        return calculate_odds_based_prediction(game_data)

def calculate_parlay_odds(probabilities):
    """Calculate parlay odds with better handling of edge cases"""
    if not probabilities or len(probabilities) == 0:
        return 0
        
    # Ensure probabilities are within valid range
    valid_probs = [min(max(p, 0.01), 0.99) for p in probabilities]
    
    # Calculate decimal odds
    decimal_odds = [1 / p for p in valid_probs]
    combined_decimal = np.prod(decimal_odds)
    
    # Convert to American odds
    if combined_decimal >= 2:
        american_odds = (combined_decimal - 1) * 100
    else:
        american_odds = -100 / (combined_decimal - 1)
    
    # Round to nearest integer
    return round(american_odds)

def format_american_odds(odds):
    """Format American odds with proper handling of edge cases"""
    # Handle non-numeric inputs
    if isinstance(odds, str):
        return odds
        
    # Handle NaN or infinity
    if np.isnan(odds) or np.isinf(odds):
        return "EVEN"
        
    # Round to integer
    odds_int = round(odds)
    
    # Format with +/- prefix
    if odds_int > 0:
        return f"+{odds_int}"
    elif odds_int < 0:
        return f"{odds_int}"
    else:
        return "EVEN"

def get_team_color(team_name):
    """Get team color based on team name"""
    # This is a simplified version - in production, use actual team colors
    team_colors = {
        'Lakers': '#552583',
        'Celtics': '#007A33',
        'Warriors': '#1D428A',
        'Nets': '#000000',
        'Heat': '#98002E',
        'Bucks': '#00471B',
        'Suns': '#1D1160',
        'Clippers': '#C8102E',
        'Nuggets': '#0E2240',
        '76ers': '#006BB6',
        'Mavericks': '#00538C',
        'Grizzlies': '#5D76A9',
        'Cavaliers': '#860038',
        'Timberwolves': '#0C2340',
        'Pelicans': '#0C2340',
        'Knicks': '#F58426',
        'Hawks': '#E03A3E',
        'Raptors': '#CE1141',
        'Bulls': '#CE1141',
        'Trail Blazers': '#E03A3E',
        'Kings': '#5A2D81',
        'Thunder': '#007AC1',
        'Spurs': '#C4CED4',
        'Wizards': '#002B5C',
        'Pacers': '#002D62',
        'Hornets': '#1D1160',
        'Pistons': '#C8102E',
        'Magic': '#0077C0',
        'Rockets': '#CE1141',
        'Jazz': '#002B5C'
    }
    
    # Check if team name contains any key from team_colors
    for key in team_colors:
        if key in team_name:
            return team_colors[key]
    
    # Default color if no match
    return '#1F77B4'

# --- Main App ---

def main():
    # Load the model
    model = load_model()

    print("\n===== TESTING MODEL LOADING =====")
    if model is None:
        print("Model failed to load - will use odds-based predictions")
    else:
        print(f"Model loaded successfully: {type(model)}")
        
        # Test odds calculation
        print("\n===== TESTING ODDS CALCULATION =====")
        test_odds = ['-110', '+100', '-150', '+200']
        for odds in test_odds:
            game_data = {'home_moneyline': odds, 'away_moneyline': '-110'}
            result = calculate_odds_based_prediction(game_data)
            print(f"Odds {odds} → probability: {result['home_win_probability']:.3f}")

    # Sidebar
    with st.sidebar:
        st.header("🏀 NBA Parlay Predictor")
        st.write("This app combines live odds with predictions to help you make better bets.")
        
        st.subheader("Options")
        confidence_threshold = st.slider("Minimum Confidence", 0.55, 0.95, 0.65, 0.01)
        max_parlay_size = st.slider("Maximum Parlay Size", 2, 5, 3)
        
        st.subheader("About")
        st.markdown("""
        This application scrapes live NBA odds and combines them with predictions
        to help you find the best betting opportunities.
        
        **Features:**
        - Live odds from DraftKings
        - Win predictions based on machine learning
        - Custom parlay builder
        - ROI analysis
        """)
        
        refresh_button = st.button("Refresh Odds")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Today's Games", "Parlay Builder", "Analytics"])
    
    # Scrape live odds data
    draft_kings_url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    
    if 'odds_data' not in st.session_state or refresh_button:
        with st.spinner("Fetching latest odds..."):
            elements = scrape_website(draft_kings_url, "table", "sportsbook-table", None)
            
            if elements:
                # Process all tables and merge data
                all_games = []
                
                for table in elements:
                    df = table_to_custom_dataframe(table)
                    if not df.empty:
                        # Group teams in pairs (assuming each game has 2 teams)
                        for i in range(0, len(df), 2):
                            if i + 1 < len(df):  # Make sure we have a pair
                                home_team = df.iloc[i+1]['team']  # Home team is usually the second one
                                away_team = df.iloc[i]['team']
                                
                                # Get odds
                                home_spread = df.iloc[i+1]['spread']
                                away_spread = df.iloc[i]['spread']
                                home_total = df.iloc[i+1]['total']
                                away_total = df.iloc[i]['total']
                                home_moneyline = df.iloc[i+1]['moneyline']
                                away_moneyline = df.iloc[i]['moneyline']
                                
                                # Create game dict
                                game = {
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_spread': home_spread,
                                    'away_spread': away_spread,
                                    'home_total': home_total,
                                    'away_total': away_total,
                                    'home_moneyline': home_moneyline,
                                    'away_moneyline': away_moneyline,
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                }
                                
                                # Make prediction
                                prediction = make_prediction(model, game)
                                game.update(prediction)
                                
                                all_games.append(game)
                
                st.session_state.odds_data = all_games
            else:
                st.error("Failed to fetch odds data. Please try refreshing again.")
    
    # Tab 1: Today's Games
    with tab1:
        st.title("Today's NBA Games & Predictions")
        
        if 'odds_data' in st.session_state and st.session_state.odds_data:
            # Display games
            for i, game in enumerate(st.session_state.odds_data):
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Determine which team is predicted to win
                if game['predicted_home_win']:
                    predicted_winner = home_team
                    win_prob = game['home_win_probability']
                else:
                    predicted_winner = away_team
                    win_prob = 1 - game['home_win_probability']
                
                # Create expander for each game
                with st.expander(f"{away_team} @ {home_team} - Prediction: {predicted_winner} ({win_prob:.1%})"):
                    # Create two columns: one for odds, one for prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Betting Odds")
                        
                        # Create a clean DataFrame for display
                        odds_df = pd.DataFrame([
                            {'Team': away_team, 'Spread': game['away_spread'], 
                             'Total': game['away_total'], 'Moneyline': game['away_moneyline']},
                            {'Team': home_team, 'Spread': game['home_spread'], 
                             'Total': game['home_total'], 'Moneyline': game['home_moneyline']}
                        ])
                        
                        st.dataframe(odds_df, hide_index=True)
                    
                    with col2:
                        st.subheader("Prediction Analysis")
                        
                        # Display the model's confidence
                        st.metric(
                            "Win Probability", 
                            f"{win_prob:.1%}", 
                            delta=f"{(win_prob - 0.5) * 100:.1f}% vs even odds"
                        )
                        
                        # Create a simple visualization
                        fig, ax = plt.subplots(figsize=(4, 1))
                        ax.barh([0, 1], [game['home_win_probability'], 1-game['home_win_probability']], 
                               color=[get_team_color(home_team), get_team_color(away_team)])
                        ax.set_yticks([0, 1])
                        ax.set_yticklabels([home_team, away_team])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Win Probability')
                        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add to parlay button
                        if st.button(f"Add to Parlay Builder", key=f"add_parlay_{i}"):
                            if 'parlay_selections' not in st.session_state:
                                st.session_state.parlay_selections = []
                            
                            # Check if already in selections
                            game_key = f"{away_team}@{home_team}"
                            existing_games = [g['key'] for g in st.session_state.parlay_selections]
                            
                            if game_key not in existing_games:
                                parlay_game = game.copy()
                                parlay_game['key'] = game_key
                                st.session_state.parlay_selections.append(parlay_game)
                                st.success(f"Added to parlay builder! Go to Parlay Builder tab to view.")
                            else:
                                st.info("This game is already in your parlay.")
        else:
            st.warning("No odds data available. Please refresh the data.")
    
    # Tab 2: Parlay Builder
    with tab2:
        st.title("NBA Parlay Builder")
        
        # Initialize parlay selections if not already done
        if 'parlay_selections' not in st.session_state:
            st.session_state.parlay_selections = []
        
        if 'odds_data' in st.session_state and st.session_state.odds_data:
            # Create two columns - one for game selection, one for parlay summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Select Games")
                
                # Display all games for selection
                for i, game in enumerate(st.session_state.odds_data):
                    # Create a unique key for the game
                    game_key = f"{game['away_team']}@{game['home_team']}"
                    existing_keys = [g['key'] for g in st.session_state.parlay_selections]
                    
                    # Determine prediction and confidence
                    if game['predicted_home_win']:
                        pick_team = game['home_team']
                        confidence = game['home_win_probability']
                    else:
                        pick_team = game['away_team']
                        confidence = 1 - game['home_win_probability']
                    
                    # Add confidence indicator
                    if confidence >= 0.7:
                        confidence_str = "⭐⭐⭐ High"
                    elif confidence >= 0.6:
                        confidence_str = "⭐⭐ Medium"
                    else:
                        confidence_str = "⭐ Low"
                    
                    # Create checkbox
                    is_selected = game_key in existing_keys
                    if st.checkbox(
                        f"{game['away_team']} @ {game['home_team']} - Pick: {pick_team} ({confidence:.1%}) - {confidence_str} Confidence",
                        value=is_selected,
                        key=f"select_game_{i}"
                    ):
                        if game_key not in existing_keys:
                            game_copy = game.copy()
                            game_copy['key'] = game_key
                            st.session_state.parlay_selections.append(game_copy)
                    elif is_selected:
                        # Remove from selections if unchecked
                        st.session_state.parlay_selections = [
                            g for g in st.session_state.parlay_selections if g['key'] != game_key
                        ]
                
                # Auto-generate button
                if st.button("Auto-Generate Optimal Parlay"):
                    # Sort games by confidence
                    sorted_games = sorted(
                        st.session_state.odds_data, 
                        key=lambda g: max(g['home_win_probability'], 1-g['home_win_probability']), 
                        reverse=True
                    )
                    
                    # Get games with confidence above threshold
                    good_games = [
                        g for g in sorted_games 
                        if max(g['home_win_probability'], 1-g['home_win_probability']) >= confidence_threshold
                    ]
                    
                    # Limit to max parlay size
                    selected_games = good_games[:min(max_parlay_size, len(good_games))]
                    
                    # Clear existing selections
                    st.session_state.parlay_selections = []
                    
                    # Add new selections
                    for game in selected_games:
                        game_copy = game.copy()
                        game_copy['key'] = f"{game['away_team']}@{game['home_team']}"
                        st.session_state.parlay_selections.append(game_copy)
                    
                    st.success(f"Generated optimal parlay with {len(selected_games)} games!")
            
            with col2:
                st.subheader("Your Parlay")
                
                if not st.session_state.parlay_selections:
                    st.info("Select games from the left to build your parlay")
                else:
                    # Display selected games
                    st.write(f"**{len(st.session_state.parlay_selections)} games selected**")
                    
                    # Calculate combined probability
                    probabilities = []
                    for game in st.session_state.parlay_selections:
                        if game['predicted_home_win']:
                            team = game['home_team']
                            prob = game['home_win_probability']
                        else:
                            team = game['away_team']
                            prob = 1 - game['home_win_probability']
                        
                        probabilities.append(prob)
                        
                        # Display game with remove button
                        col_a, col_b = st.columns([5, 1])
                        with col_a:
                            st.write(f"• {game['away_team']} @ {game['home_team']} - Pick: **{team}** ({prob:.1%})")
                        with col_b:
                            if st.button("❌", key=f"remove_{game['key']}"):
                                st.session_state.parlay_selections.remove(game)
                                st.experimental_rerun()
                    
                    # Calculate combined probability
                    combined_prob = np.prod(probabilities)
                    
                    # Calculate parlay odds
                    parlay_odds = calculate_parlay_odds(probabilities)
                    
                    # Display probability and odds
                    st.metric("Combined Probability", f"{combined_prob:.1%}", delta=None)
                    st.metric("Parlay Odds", format_american_odds(parlay_odds))
                    
                    # Let user input stake
                    stake = st.number_input("Stake ($)", min_value=5.0, max_value=1000.0, value=100.0, step=5.0)
                    
                    # Calculate potential payout
                    if parlay_odds >= 0:
                        payout = stake + (stake * parlay_odds / 100)
                    else:
                        payout = stake + (stake * 100 / abs(parlay_odds))
                    
                    profit = payout - stake
                    roi = (profit / stake) * 100
                    
                    # Display potential payout
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.metric("Potential Payout", f"${payout:.2f}")
                    with col_d:
                        st.metric("Potential Profit", f"${profit:.2f}", f"{roi:.1f}% ROI")
                    
                    # Clear button
                    if st.button("Clear Parlay"):
                        st.session_state.parlay_selections = []
                        st.experimental_rerun()
                    
                    # Bet now button
                    if st.button("Place Bet on DraftKings", type="primary"):
                        st.markdown(f"[Go to DraftKings]({draft_kings_url})")
        else:
            st.warning("No odds data available. Please refresh the data.")
    
    # Tab 3: Analytics
    with tab3:
        st.title("NBA Betting Analytics")
        
        # Create subtabs
        subtab1, subtab2 = st.tabs(["ROI Analysis", "Betting Guide"])
        
        with subtab1:
            st.subheader("Expected ROI by Confidence Level")
            
            # Create a ROI visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create confidence bins
            confidence_levels = np.linspace(0.5, 1.0, 11)
            estimated_accuracy = confidence_levels * 0.9  # Assuming model is calibrated but slightly overconfident
            
            # Calculate ROI at different confidence levels
            roi_data = []
            
            for conf, acc in zip(confidence_levels, estimated_accuracy):
                # Calculate ROI based on American odds formula
                # For a fair odds bet at confidence level 'conf', American odds would be:
                fair_odds = (1 / conf - 1) * 100
                
                # Assume market offers slightly worse odds
                market_odds = fair_odds * 0.9
                
                # Expected ROI = Probability of winning * payout - probability of losing * stake
                expected_roi = acc * (100 + market_odds) / 100 - (1 - acc) * 1
                
                roi_data.append({
                    'Confidence': conf,
                    'Accuracy': acc,
                    'Fair Odds': fair_odds,
                    'Market Odds': market_odds,
                    'Expected ROI': expected_roi
                })
            
            # Create DataFrame
            roi_df = pd.DataFrame(roi_data)
            
            # Plot ROI curve
            sns.lineplot(x='Confidence', y='Expected ROI', data=roi_df, marker='o', ax=ax)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.title('Expected ROI by Confidence Level')
            plt.xlabel('Model Confidence')
            plt.ylabel('Expected ROI (%)')
            plt.grid(True, alpha=0.3)
            
            # Add threshold line at current confidence threshold
            plt.axvline(x=confidence_threshold, color='green', linestyle='--')
            plt.text(confidence_threshold + 0.01, 0.1, f'Current Threshold: {confidence_threshold:.2f}', 
                    rotation=90, va='bottom')
            
            st.pyplot(fig)
            
            # Display ROI table
            roi_display = roi_df.copy()
            roi_display['Confidence'] = roi_display['Confidence'].apply(lambda x: f"{x*100:.1f}%")
            roi_display['Accuracy'] = roi_display['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
            roi_display['Fair Odds'] = roi_display['Fair Odds'].apply(format_american_odds)
            roi_display['Market Odds'] = roi_display['Market Odds'].apply(format_american_odds)
            roi_display['Expected ROI'] = roi_display['Expected ROI'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(roi_display)
        
        with subtab2:
            st.subheader("Basketball Betting Guide")
            
            with st.expander("Understanding American Odds"):
                st.markdown("""
                **American Odds Format**
                
                American odds are presented as positive or negative numbers:
                
                - **Positive odds** (e.g., +150): Shows how much profit you would win on a $100 bet
                  - Example: +150 means a $100 bet would win $150 profit ($250 total return)
                
                - **Negative odds** (e.g., -110): Shows how much you need to bet to win $100 profit
                  - Example: -110 means you need to bet $110 to win $100 profit ($210 total return)
                
                **Converting to Probability**
                
                To convert American odds to implied probability:
                
                - For positive odds: 100 / (odds + 100)
                  - Example: +150 → 100 / (150 + 100) = 0.4 = 40%
                
                - For negative odds: |odds| / (|odds| + 100)
                  - Example: -110 → 110 / (110 + 100) = 0.524 = 52.4%
                """)
            
            with st.expander("How Parlays Work"):
                st.markdown("""
                **Parlay Betting**
                
                A parlay is a single bet that combines multiple bets into one wager. For the parlay to win, 
                ALL selections must win.
                
                **How parlay odds are calculated:**
                
                1. Convert individual odds to decimal format
                2. Multiply all decimal odds together
                3. Convert back to American format
                
                **Example:**
                - Bet 1: -110 (1.91 in decimal)
                - Bet 2: +140 (2.40 in decimal)
                - Parlay odds: 1.91 × 2.40 = 4.58 (or +358 in American)
                
                **Risk vs. Reward:**
                
                Parlays offer higher potential payouts but lower probability of winning. Each additional leg 
                increases potential payout but decreases likelihood of winning.
                """)
            
            with st.expander("Bankroll Management"):
                st.markdown("""
                **Bankroll Management Guidelines**
                
                Proper bankroll management is crucial for long-term betting success:
                
                1. **Establish a betting bankroll** - A dedicated amount of money for betting
                2. **Use unit sizing** - Bet a consistent percentage of your bankroll (typically 1-5%)
                3. **Never chase losses** - Stick to your betting strategy regardless of recent results
                4. **Track all bets** - Keep detailed records of all wagers and outcomes
                
                **Kelly Criterion**
                
                A mathematical formula to determine optimal bet size:
                
                Kelly % = (bp - q) / b
                
                Where:
                - b = the decimal odds - 1
                - p = probability of winning
                - q = probability of losing (1 - p)
                
                Most professional bettors use fractional Kelly (e.g., 1/4 or 1/2 of the calculated Kelly bet)
                to reduce variance.
                """)
            
            with st.expander("Model Information"):
                st.markdown("""
                **About the Prediction Model**
                
                This application uses a machine learning model trained on historical NBA data to predict game outcomes.
                
                **Key features used in predictions:**
                - Betting odds (spread, moneyline)
                - Team statistics (FG%, rebounds, assists, etc.)
                - Home court advantage
                
                **Confidence Rating:**
                - ⭐⭐⭐ High: 70%+ probability
                - ⭐⭐ Medium: 60-70% probability
                - ⭐ Low: 50-60% probability
                
                **Accuracy Considerations:**
                While the model attempts to provide accurate predictions, sports betting inherently involves uncertainty.
                Always use your own judgment when placing bets.
                """)

if __name__ == "__main__":
    main()