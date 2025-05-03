"""
NBA Data Demo - Retrieving real NBA data for Parlay Prediction
"""
import pandas as pd
import numpy as np
import time
import os
from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats
from nba_api.stats.static import teams

def get_team_data():
    """Get all NBA teams and their IDs"""
    # Get all NBA teams
    nba_teams = teams.get_teams()
    print(f"Found {len(nba_teams)} NBA teams")
    
    # Display first few teams
    for team in nba_teams[:5]:
        print(f"Team: {team['full_name']}, ID: {team['id']}")
    
    return nba_teams

def get_games_data(season='2023-24', max_games=None):
    """Get game data for a specific season"""
    print(f"Fetching game data for {season} season...")
    
    try:
        # Use NBA API to get game data
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',  # NBA league ID
            season_type_nullable='Regular Season'
        )
        
        # Convert to DataFrame
        games_df = gamefinder.get_data_frames()[0]
        
        # Limit number of games if specified
        if max_games is not None:
            games_df = games_df.head(max_games)
        
        # Display basic info
        print(f"Retrieved {len(games_df)} game records")
        print("\nSample of game data:")
        print(games_df[['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head())
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        output_file = f"data/raw/nba_games_{season.replace('-', '_')}.csv"
        games_df.to_csv(output_file, index=False)
        print(f"Saved game data to {output_file}")
        
        return games_df
    except Exception as e:
        print(f"Error fetching game data: {e}")
        return None

def get_team_stats(season='2023-24'):
    """Get advanced team statistics for a specific season"""
    print(f"Fetching advanced team stats for {season} season...")
    
    try:
        # Use NBA API to get advanced team stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        
        # Add a delay to respect API rate limits
        time.sleep(1)
        
        # Convert to DataFrame
        stats_df = team_stats.get_data_frames()[0]
        
        # Display basic info
        print(f"Retrieved stats for {len(stats_df)} teams")
        print("\nSample of team stats:")
        
        # Select a subset of columns for display
        display_cols = ['TEAM_NAME', 'W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']
        print(stats_df[display_cols].head())
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        output_file = f"data/raw/nba_team_stats_{season.replace('-', '_')}.csv"
        stats_df.to_csv(output_file, index=False)
        print(f"Saved team stats to {output_file}")
        
        return stats_df
    except Exception as e:
        print(f"Error fetching team stats: {e}")
        return None

def create_betting_features(games_df, team_stats_df):
    """
    Create betting-related features by combining game data and team stats
    This is just an example - in a real application, you'd get real odds data
    """
    print("Creating synthetic betting features based on real team performance...")
    
    # Ensure we have valid data
    if games_df is None or team_stats_df is None:
        print("Error: Cannot create betting features due to missing data.")
        return None
    
    if len(games_df) == 0 or len(team_stats_df) == 0:
        print("Error: Empty data frames provided.")
        return None
    
    # Merge team stats with game data
    games_with_stats = []
    
    # Try to find required columns
    required_columns = ['TEAM_NAME', 'MATCHUP', 'GAME_ID', 'WL', 'PTS', 'GAME_DATE']
    missing_columns = [col for col in required_columns if col not in games_df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns in games data: {missing_columns}")
        return None
    
    required_stats_columns = ['TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']
    missing_stats_columns = [col for col in required_stats_columns if col not in team_stats_df.columns]
    
    if missing_stats_columns:
        print(f"Error: Missing required columns in team stats data: {missing_stats_columns}")
        return None
    
    # Normalize team names if needed
    print(f"Mapping team names between datasets...")
    
    # Create a mapping for team names that might be different between datasets
    nba_teams_data = teams.get_teams()
    team_name_mapping = {}
    
    for team in nba_teams_data:
        # Add various forms of team names
        full_name = team['full_name']
        city = team['city']
        nickname = team['nickname']
        
        team_name_mapping[full_name] = full_name
        team_name_mapping[city] = full_name
        team_name_mapping[nickname] = full_name
        
        # Special cases
        if 'LA' in city:
            team_name_mapping['Los Angeles ' + nickname] = full_name
    
    # Process game data and enrich with team stats
    print("Processing game data...")
    
    for _, game in games_df.iterrows():
        team_name = game['TEAM_NAME']
        
        # Try to map team name if needed
        if team_name in team_name_mapping:
            team_name = team_name_mapping[team_name]
        
        # Find team stats
        team_stats = team_stats_df[team_stats_df['TEAM_NAME'] == team_name]
        
        # If not found, try alternative team name forms
        if len(team_stats) == 0:
            for alt_name, mapped_name in team_name_mapping.items():
                if alt_name in team_name:
                    team_stats = team_stats_df[team_stats_df['TEAM_NAME'] == mapped_name]
                    if len(team_stats) > 0:
                        break
        
        if len(team_stats) > 0:
            game_dict = game.to_dict()
            
            # Add team rating info
            game_dict['OFF_RATING'] = team_stats['OFF_RATING'].values[0]
            game_dict['DEF_RATING'] = team_stats['DEF_RATING'].values[0]
            game_dict['NET_RATING'] = team_stats['NET_RATING'].values[0]
            
            games_with_stats.append(game_dict)
    
    print(f"Processed {len(games_with_stats)} games with team stats")
    
    # Check if we have enough data
    if len(games_with_stats) == 0:
        print("Error: No games could be matched with team stats.")
        return None
    
    # Convert to DataFrame
    enhanced_df = pd.DataFrame(games_with_stats)
    
    # Create synthetic odds for home games
    print("Generating synthetic betting odds...")
    home_games = enhanced_df[enhanced_df['MATCHUP'].str.contains('vs.', na=False)].copy()
    
    if len(home_games) == 0:
        print("Error: No home games found in the data.")
        return None
    
    print(f"Found {len(home_games)} home games for odds generation")
    
    odds_data = []
    for _, game in home_games.iterrows():
        home_team = game['TEAM_NAME']
        
        try:
            # Extract away team from matchup
            matchup = game['MATCHUP']
            if isinstance(matchup, str) and 'vs.' in matchup:
                # Different matchup formats possible
                if '@' in matchup:
                    # Format: "TeamA @ TeamB"
                    parts = matchup.split('@')
                    if len(parts) == 2:
                        if home_team in parts[1]:
                            away_team = parts[0].strip()
                        else:
                            away_team = parts[1].strip()
                else:
                    # Format: "TeamA vs. TeamB"
                    parts = matchup.split('vs.')
                    if len(parts) == 2:
                        if home_team in parts[0]:
                            away_team = parts[1].strip()
                        else:
                            away_team = parts[0].strip()
            else:
                # Skip if matchup format is unexpected
                continue
            
            # Find the away team's game record
            away_games = enhanced_df[
                (enhanced_df['GAME_ID'] == game['GAME_ID']) & 
                (enhanced_df['TEAM_NAME'] != home_team)
            ]
            
            if len(away_games) == 0:
                # Try matching by date and teams
                game_date = game['GAME_DATE']
                away_games = enhanced_df[
                    (enhanced_df['GAME_DATE'] == game_date) & 
                    (enhanced_df['TEAM_NAME'] != home_team)
                ]
            
            # Skip if we can't find the away team's record
            if len(away_games) == 0:
                continue
                
            # Use the first matching away game
            away_game = away_games.iloc[0]
            away_team = away_game['TEAM_NAME']
            
            # Get ratings for odds calculation
            home_net_rating = game['NET_RATING']
            away_net_rating = away_game['NET_RATING']
            
            # Calculate rating difference
            rating_diff = home_net_rating - away_net_rating
            
            # Generate synthetic odds based on rating difference
            if rating_diff > 3:  # Home team clear favorite
                home_odds = np.random.uniform(-190, -140)
                away_odds = np.random.uniform(120, 180)
                spread = np.random.uniform(-6.5, -3.5)
            elif rating_diff > 0:  # Home team slight favorite
                home_odds = np.random.uniform(-140, -110)
                away_odds = np.random.uniform(100, 130)
                spread = np.random.uniform(-3.5, -1.5)
            elif rating_diff > -3:  # Away team slight favorite
                home_odds = np.random.uniform(100, 130)
                away_odds = np.random.uniform(-140, -110)
                spread = np.random.uniform(1.5, 3.5)
            else:  # Away team clear favorite
                home_odds = np.random.uniform(120, 180)
                away_odds = np.random.uniform(-190, -140)
                spread = np.random.uniform(3.5, 6.5)
                
            # Add to odds data
            odds_data.append({
                'GAME_DATE': game['GAME_DATE'],
                'GAME_ID': game['GAME_ID'],
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'HOME_PTS': game['PTS'],
                'HOME_NET_RATING': home_net_rating,
                'AWAY_NET_RATING': away_net_rating,
                'RATING_DIFF': rating_diff,
                'HOME_ODDS': home_odds,
                'AWAY_ODDS': away_odds,
                'SPREAD': spread,
                'HOME_WIN': 1 if game['WL'] == 'W' else 0
            })
        except Exception as e:
            print(f"Error processing game {game['GAME_ID']}: {e}")
    
    # Check if we have odds data
    if len(odds_data) == 0:
        print("Error: Failed to generate betting odds.")
        return None
    
    # Convert to DataFrame
    odds_df = pd.DataFrame(odds_data)
    
    # Display sample
    print(f"\nGenerated betting odds for {len(odds_df)} games")
    print("\nSample of generated betting data:")
    display_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'RATING_DIFF', 'HOME_ODDS', 'AWAY_ODDS', 'SPREAD', 'HOME_WIN']
    print(odds_df[display_cols].head())
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    output_file = f"data/raw/betting_odds.csv"
    odds_df.to_csv(output_file, index=False)
    print(f"Saved betting odds to {output_file}")
    
    return odds_df

def get_player_performance_data(season='2023-24', max_players=50):
    """
    Get individual player performance data to enhance predictions
    This would be used for more advanced features in the model
    """
    print("This function would retrieve player performance data for advanced models")
    print("Not implemented in this demo")
    # In a full implementation, you would use endpoints like:
    # - playercareerstats
    # - playerprofilev2
    # - leaguedashplayerstats

def run_demo(season='2023-24'):
    """Run the complete data retrieval demo"""
    print(f"NBA Data Retrieval Demo for {season} Season")
    print("=" * 50)
    
    # 1. Get team data
    teams_data = get_team_data()
    
    # 2. Get games data (limit to 500 games for demo)
    games_df = get_games_data(season, max_games=500)
    if games_df is None:
        print("Error: Failed to retrieve game data. Demo aborted.")
        return
    
    # 3. Get team stats
    team_stats_df = get_team_stats(season)
    if team_stats_df is None:
        print("Error: Failed to retrieve team stats. Demo aborted.")
        return
    
    # 4. Create betting features
    odds_df = create_betting_features(games_df, team_stats_df)
    if odds_df is None:
        print("Error: Failed to create betting features. Demo completed with partial results.")
    
    print("=" * 50)
    print("Demo completed!")
    print("The data can now be used with the NBA Parlay Predictor model.")

if __name__ == "__main__":
    # Uncomment to specify a different season
    # run_demo(season='2022-23')
    
    run_demo()