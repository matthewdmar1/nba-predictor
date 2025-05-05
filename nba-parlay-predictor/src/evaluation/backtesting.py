import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from src.models.train_model import train_model
from src.models.evaluate import generate_parlays

class BacktestEngine:
    """Backtest engine for NBA parlay predictions"""
    
    def __init__(self, data, target_column, feature_columns, date_column,
                min_confidence=0.65, max_games=3, stake=100, window_size=60):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        data : pandas DataFrame
            Full dataset with timestamp column
        target_column : str
            Name of the target column
        feature_columns : list
            List of feature column names
        date_column : str
            Name of the date column
        min_confidence : float
            Minimum confidence threshold for including games in parlays
        max_games : int
            Maximum number of games per parlay
        stake : float
            Stake amount per parlay
        window_size : int
            Number of days in training window
        """
        self.data = data.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.date_column = date_column
        self.min_confidence = min_confidence
        self.max_games = max_games
        self.stake = stake
        self.window_size = window_size
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            
        # Sort data by date
        self.data = self.data.sort_values(date_column).reset_index(drop=True)
    
    def run_backtest(self, start_date=None, end_date=None, forward_test_period=None):
        """
        Run backtesting simulation
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for backtest (default: min date + window_size)
        end_date : str or datetime
            End date for backtest (default: max date)
        forward_test_period : int, optional
            Number of days to hold out for forward testing
            
        Returns:
        --------
        results : dict
            Dictionary with backtest results
        """
        # Convert dates if needed
        if start_date is None:
            min_date = self.data[self.date_column].min()
            start_date = min_date + timedelta(days=self.window_size)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            end_date = self.data[self.date_column].max()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # If forward testing is requested
        if forward_test_period is not None:
            print(f"\nRunning forward test on completely held-out {forward_test_period} days...")
            
            # Define forward test period (last n days of data)
            forward_start = end_date - timedelta(days=forward_test_period)
            
            # Ensure no data from forward period is used for backtest parameter tuning
            backtest_end = forward_start - timedelta(days=1)
            
            # Run backtest only up to the forward test cutoff
            backtest_results = self._run_backtest_period(start_date, backtest_end)
            
            # Now test on forward period with frozen parameters
            forward_results = self._run_forward_test(forward_start, end_date, backtest_results)
            
            return {
                'backtest_results': backtest_results,
                'forward_test_results': forward_results
            }
        else:
            # Normal backtest
            return self._run_backtest_period(start_date, end_date)
    
    def _run_backtest_period(self, start_date, end_date):
        """Run backtest for a specific period"""
        # Get all unique dates in the test period
        test_dates = self.data[
            (self.data[self.date_column] >= start_date) & 
            (self.data[self.date_column] <= end_date)
        ][self.date_column].drop_duplicates().sort_values()
        
        # Prepare results containers
        bets = []
        daily_results = []
        
        # Run backtest day by day
        for current_date in test_dates:
            # Define training window
            window_start = current_date - timedelta(days=self.window_size)
            
            # Get training data
            train_data = self.data[
                (self.data[self.date_column] >= window_start) & 
                (self.data[self.date_column] < current_date)
            ]
            
            # Get test data (today's games)
            test_data = self.data[self.data[self.date_column] == current_date]
            
            # Skip if not enough data
            if len(train_data) < 50 or len(test_data) == 0:
                continue
                
            # Extract features and target
            X_train = train_data[self.feature_columns]
            y_train = train_data[self.target_column]
            X_test = test_data[self.feature_columns]
            
            # Keep team information for reference
            teams_info = test_data[[
                self.date_column, 'HOME_TEAM', 'AWAY_TEAM', 
                'HOME_PTS', 'AWAY_PTS', 'HOME_WIN'
            ]]
            
            # Train model on rolling window
            model = train_model(X_train, y_train, cv_folds=3)
            
            # Make predictions
            y_pred = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            # Create results dataframe
            results_df = teams_info.copy()
            results_df['predicted_home_win'] = y_pred
            results_df['home_win_probability'] = probabilities
            results_df['correct_prediction'] = (results_df['HOME_WIN'] == results_df['predicted_home_win'])
            
            # Generate parlays
            parlays = generate_parlays(
                results_df, 
                min_prob=self.min_confidence, 
                max_games=self.max_games
            )
            
            # Record bets and outcomes
            day_profit = 0
            day_stake = 0
            day_payout = 0
            
            for parlay in parlays:
                # Calculate potential payout based on American odds
                probabilities = [
                    g['home_win_probability'] if g['predicted_home_win'] else (1 - g['home_win_probability'])
                    for g in parlay['games']
                ]
                
                decimal_odds = [1 / p for p in probabilities]
                combined_decimal = np.prod(decimal_odds)
                american_odds = (combined_decimal - 1) * 100
                
                if american_odds >= 0:
                    payout = self.stake + (self.stake * american_odds / 100)
                else:
                    payout = self.stake + (self.stake * 100 / abs(american_odds))
                
                # Check if parlay won (all predictions correct)
                all_correct = True
                for game in parlay['games']:
                    if game['predicted_home_win'] != game['HOME_WIN']:
                        all_correct = False
                        break
                
                # Record results
                profit = payout - self.stake if all_correct else -self.stake
                
                bets.append({
                    'date': current_date,
                    'parlay_size': parlay['parlay_size'],
                    'combined_probability': parlay['combined_probability'],
                    'american_odds': american_odds,
                    'stake': self.stake,
                    'potential_payout': payout,
                    'won': all_correct,
                    'actual_payout': payout if all_correct else 0,
                    'profit': profit,
                    'games': '; '.join([
                        f"{g['HOME_TEAM']} vs {g['AWAY_TEAM']}: {'HOME' if g['predicted_home_win'] else 'AWAY'}"
                        for g in parlay['games']
                    ])
                })
                
                day_profit += profit
                day_stake += self.stake
                day_payout += payout if all_correct else 0
            
            # Record daily results
            if parlays:
                daily_results.append({
                    'date': current_date,
                    'num_parlays': len(parlays),
                    'stake': day_stake,
                    'payout': day_payout,
                    'profit': day_profit,
                    'roi': (day_profit / day_stake * 100) if day_stake > 0 else 0
                })
        
        # Convert to DataFrames
        bets_df = pd.DataFrame(bets)
        daily_df = pd.DataFrame(daily_results)
        
        # Calculate cumulative results
        if len(daily_df) > 0:
            daily_df['cumulative_profit'] = daily_df['profit'].cumsum()
            daily_df['cumulative_roi'] = daily_df['cumulative_profit'] / (daily_df['stake'].cumsum()) * 100
        
        return {
            'bets': bets_df,
            'daily_results': daily_df,
            'total_bets': len(bets_df),
            'win_rate': bets_df['won'].mean() if len(bets_df) > 0 else 0,
            'total_profit': bets_df['profit'].sum() if len(bets_df) > 0 else 0,
            'total_roi': bets_df['profit'].sum() / bets_df['stake'].sum() * 100 if len(bets_df) > 0 and bets_df['stake'].sum() > 0 else 0
        }
    
    def _run_forward_test(self, start_date, end_date, backtest_results):
        """Run forward test using parameters from backtest"""
        # Implement forward testing logic here
        # For now, this is just a placeholder
        # You'll want to use the same parameters from backtest_results
        # but apply them to the forward test period
        print(f"Running forward test from {start_date} to {end_date}")
        
        # Similar logic to _run_backtest_period, but using models/parameters from backtest
        # For this example, we'll just run a similar backtest
        return self._run_backtest_period(start_date, end_date)
    
    def plot_results(self, results):
        """Plot backtest results"""
        # Check if we have enough data to plot
        if len(results['daily_results']) == 0:
            return "No results to plot"
            
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot cumulative profit over time
        sns.lineplot(
            data=results['daily_results'],
            x='date',
            y='cumulative_profit',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Cumulative Profit Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Plot daily profit
        sns.barplot(
            data=results['daily_results'].tail(20),  # Show last 20 days for readability
            x='date',
            y='profit',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Daily Profit (Last 20 Days)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Profit ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot win rate by parlay size
        if len(results['bets']) > 0:
            win_by_size = results['bets'].groupby('parlay_size')['won'].agg(['mean', 'count']).reset_index()
            win_by_size.columns = ['parlay_size', 'win_rate', 'count']
            
            sns.barplot(
                data=win_by_size,
                x='parlay_size',
                y='win_rate',
                ax=axes[1, 0]
            )
            axes[1, 0].set_title('Win Rate by Parlay Size')
            axes[1, 0].set_xlabel('Parlay Size')
            axes[1, 0].set_ylabel('Win Rate')
            
            # Add count labels
            for i, row in win_by_size.iterrows():
                axes[1, 0].text(
                    i, row['win_rate'] + 0.02, 
                    f"n={int(row['count'])}", 
                    ha='center'
                )
        
        # Plot probability calibration
        if len(results['bets']) > 0:
            # Create probability bins
            results['bets']['prob_bin'] = pd.cut(
                results['bets']['combined_probability'], 
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            
            # Calculate actual win rate per bin
            calibration = results['bets'].groupby('prob_bin')['won'].agg(['mean', 'count']).reset_index()
            calibration.columns = ['prob_bin', 'actual_win_rate', 'count']
            
            # Get bin centers for plotting
            calibration['bin_center'] = calibration['prob_bin'].apply(lambda x: (x.left + x.right) / 2)
            
            # Plot calibration curve
            sns.scatterplot(
                data=calibration,
                x='bin_center',
                y='actual_win_rate',
                size='count',
                sizes=(20, 200),
                ax=axes[1, 1]
            )
            
            # Add perfect calibration line
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            axes[1, 1].set_title('Probability Calibration')
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].set_ylabel('Actual Win Rate')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
        
        # Add overall stats as text
        fig.suptitle(f"Backtest Results Summary: {results['win_rate']:.1%} Win Rate, ${results['total_profit']:.2f} Profit, {results['total_roi']:.1f}% ROI", 
                    fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        return fig