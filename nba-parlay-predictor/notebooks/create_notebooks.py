import nbformat as nbf
import os

def create_notebook(cells_content, output_path):
    """
    Create a Jupyter notebook with provided cell content
    
    Parameters:
    -----------
    cells_content : list of dict
        List of cell content dictionaries with 'type' and 'content' keys
        'type' should be 'markdown' or 'code'
    output_path : str
        Path to save the notebook
    """
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add cells
    nb.cells = []
    for cell in cells_content:
        if cell['type'] == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(cell['content']))
        elif cell['type'] == 'code':
            nb.cells.append(nbf.v4.new_code_cell(cell['content']))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the notebook to file
    with open(output_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created at {output_path}")

# Define cells for data exploration notebook
data_exploration_cells = [
    {'type': 'markdown', 'content': '# NBA Parlay Betting Analysis - Data Exploration\n\nThis notebook explores the NBA game data and betting odds to understand patterns that might be useful for parlay betting.'},
    {'type': 'code', 'content': '# Import necessary libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set plot style\nsns.set_style(\'whitegrid\')\nplt.rcParams[\'figure.figsize\'] = (12, 8)'},
    # Add more cells from the notebook content I provided earlier
    # ...
]

# Define cells for model evaluation notebook
model_evaluation_cells = [
{
    "cells": [
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# NBA Parlay Betting Analysis - Model Evaluation\n",
        "\n",
        "This notebook evaluates different models for NBA game predictions and parlay generation."
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Import necessary libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
        "\n",
        "# Set plot style\n",
        "sns.set_style('whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (12, 8)"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Load Processed Data"
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Add the src directory to the path so we can import our modules\n",
        "import sys\n",
        "import os\n",
        "sys.path.append('..')\n",
        "\n",
        "# Import our config\n",
        "import config\n",
        "\n",
        "# Load the processed data\n",
        "try:\n",
        "    feature_df = pd.read_csv(f\"{config.DATA_PROCESSED_DIR}/features.csv\")\n",
        "    print(\"Loaded processed data successfully!\")\n",
        "except FileNotFoundError:\n",
        "    print(\"Processed data not found. Run the data processing steps first.\")\n",
        "    # Alternatively, load and process the data here\n",
        "    from src.data.fetch_data import load_nba_data, load_odds_data, generate_synthetic_data\n",
        "    from src.data.preprocess import preprocess_data\n",
        "    from src.features.build_features import engineer_features\n",
        "    \n",
        "    try:\n",
        "        games_df = load_nba_data()\n",
        "        odds_df = load_odds_data()\n",
        "    except FileNotFoundError:\n",
        "        games_df, odds_df = generate_synthetic_data()\n",
        "        \n",
        "    processed_df = preprocess_data(games_df, odds_df)\n",
        "    feature_df = engineer_features(processed_df)\n",
        "    \n",
        "    # Make sure the directory exists\n",
        "    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)\n",
        "    feature_df.to_csv(f\"{config.DATA_PROCESSED_DIR}/features.csv\", index=False)\n",
        "    print(\"Generated and saved processed data.\")"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Prepare Data for Modeling"
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Prepare features and target\n",
        "# Exclude non-feature columns\n",
        "non_feature_cols = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']\n",
        "feature_cols = [col for col in feature_df.columns if col not in non_feature_cols]\n",
        "\n",
        "X = feature_df[feature_cols]\n",
        "y = feature_df['HOME_WIN']\n",
        "\n",
        "# Keep team info for reference\n",
        "teams_info = feature_df[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS', 'HOME_WIN']]\n",
        "\n",
        "# Split the data\n",
        "X_train, X_test, y_train, y_test, teams_info_train, teams_info_test = train_test_split(\n",
        "    X, y, teams_info, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE\n",
        ")\n",
        "\n",
        "print(f\"Training set: {X_train.shape[0]} samples\")\n",
        "print(f\"Testing set: {X_test.shape[0]} samples\")\n",
        "print(f\"Features: {X_train.shape[1]}\")"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Train Models\n",
        "\n",
        "Let's train and compare different models."
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Import our model training function\n",
        "from src.models.train_model import train_model\n",
        "\n",
        "# Train logistic regression (our baseline model)\n",
        "lr_model = train_model(X_train, y_train, cv_folds=config.CV_FOLDS)\n",
        "\n",
        "# For comparison, let's also try a random forest model\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "rf_pipeline = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('classifier', RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE))\n",
        "])\n",
        "\n",
        "rf_pipeline.fit(X_train, y_train)\n",
        "rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=config.CV_FOLDS)\n",
        "print(f\"Random Forest CV accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}\")"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Evaluate Models"
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Import our evaluation functions\n",
        "from src.models.evaluate import evaluate_model\n",
        "from src.visualization.visualize import plot_confusion_matrix\n",
        "\n",
        "# Evaluate logistic regression\n",
        "print(\"\\nLogistic Regression Results:\")\n",
        "lr_results = evaluate_model(lr_model, X_test, y_test, teams_info_test)\n",
        "\n",
        "# Evaluate random forest\n",
        "print(\"\\nRandom Forest Results:\")\n",
        "rf_predictions = rf_pipeline.predict(X_test)\n",
        "rf_probabilities = rf_pipeline.predict_proba(X_test)[:, 1]\n",
        "\n",
        "accuracy = accuracy_score(y_test, rf_predictions)\n",
        "precision = precision_score(y_test, rf_predictions)\n",
        "recall = recall_score(y_test, rf_predictions)\n",
        "f1 = f1_score(y_test, rf_predictions)\n",
        "\n",
        "print(f\"Test accuracy: {accuracy:.4f}\")\n",
        "print(f\"Test precision: {precision:.4f}\")\n",
        "print(f\"Test recall: {recall:.4f}\")\n",
        "print(f\"Test F1 score: {f1:.4f}\")\n",
        "\n",
        "# Plot confusion matrices\n",
        "plt.figure(figsize=(12, 5))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "plot_confusion_matrix(y_test, lr_results['predicted_home_win'])\n",
        "plt.title('Logistic Regression Confusion Matrix')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plot_confusion_matrix(y_test, rf_predictions)\n",
        "plt.title('Random Forest Confusion Matrix')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Compare Model Calibration\n",
        "\n",
        "Let's see how well calibrated our models are by comparing predicted probabilities to actual outcomes."
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Create dataframes with predictions\n",
        "lr_df = pd.DataFrame({\n",
        "    'probability': lr_model.predict_proba(X_test)[:, 1],\n",
        "    'prediction': lr_model.predict(X_test),\n",
        "    'actual': y_test,\n",
        "    'model': 'Logistic Regression'\n",
        "})\n",
        "\n",
        "rf_df = pd.DataFrame({\n",
        "    'probability': rf_pipeline.predict_proba(X_test)[:, 1],\n",
        "    'prediction': rf_pipeline.predict(X_test),\n",
        "    'actual': y_test,\n",
        "    'model': 'Random Forest'\n",
        "})\n",
        "\n",
        "# Combine dataframes\n",
        "combined_df = pd.concat([lr_df, rf_df])\n",
        "\n",
        "# Create bins for probability\n",
        "bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]\n",
        "combined_df['prob_bin'] = pd.cut(combined_df['probability'], bins=bins)\n",
        "\n",
        "# Calculate actual win rates by predicted probability bin\n",
        "calibration = combined_df.groupby(['model', 'prob_bin'])['actual'].agg(['mean', 'count'])\n",
        "calibration.columns = ['actual_win_rate', 'count']\n",
        "calibration = calibration.reset_index()\n",
        "\n",
        "# Create bin centers for plotting\n",
        "bin_centers = [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9]\n",
        "calibration['bin_center'] = calibration['prob_bin'].cat.codes.map(dict(enumerate(bin_centers)))\n",
        "\n",
        "# Plot calibration curves\n",
        "plt.figure(figsize=(10, 8))\n",
        "\n",
        "for model_name, color in zip(['Logistic Regression', 'Random Forest'], ['blue', 'green']):\n",
        "    model_data = calibration[calibration['model'] == model_name]\n",
        "    \n",
        "    # Plot predicted vs actual probabilities\n",
        "    plt.scatter(model_data['bin_center'], model_data['actual_win_rate'], \n",
        "               s=model_data['count'], alpha=0.7, color=color, label=model_name)\n",
        "    \n",
        "    # Connect points with lines\n",
        "    plt.plot(model_data['bin_center'], model_data['actual_win_rate'], color=color, alpha=0.5)\n",
        "\n",
        "# Add a perfect calibration line\n",
        "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')\n",
        "\n",
        "plt.title('Model Calibration Comparison')\n",
        "plt.xlabel('Predicted Probability')\n",
        "plt.ylabel('Actual Win Rate')\n",
        "plt.xlim(0, 1)\n",
        "plt.ylim(0, 1)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Generate Parlay Recommendations"
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Import our parlay generation function\n",
        "from src.models.evaluate import generate_parlays, simulate_roi\n",
        "\n",
        "# Create results dataframe for random forest model\n",
        "rf_results = teams_info_test.copy()\n",
        "rf_results['predicted_home_win'] = rf_predictions\n",
        "rf_results['home_win_probability'] = rf_probabilities\n",
        "rf_results['correct_prediction'] = (rf_results['HOME_WIN'] == rf_results['predicted_home_win'])\n",
        "\n",
        "# Generate parlays using logistic regression\n",
        "print(\"\\nLogistic Regression Parlay Recommendations:\")\n",
        "lr_parlays = generate_parlays(lr_results, min_prob=config.MIN_CONFIDENCE, max_games=config.MAX_PARLAY_SIZE)\n",
        "\n",
        "# Generate parlays using random forest\n",
        "print(\"\\nRandom Forest Parlay Recommendations:\")\n",
        "rf_parlays = generate_parlays(rf_results, min_prob=config.MIN_CONFIDENCE, max_games=config.MAX_PARLAY_SIZE)\n",
        "\n",
        "# Display parlay recommendations\n",
        "def display_parlays(parlays, model_name):\n",
        "    print(f\"\\n{model_name} Parlay Recommendations:\")\n",
        "    for i, parlay in enumerate(parlays):\n",
        "        print(f\"\\nParlay #{i+1}:\")\n",
        "        print(f\"Combined probability: {parlay['combined_probability']:.4f}\")\n",
        "        print(f\"Number of games: {parlay['parlay_size']}\")\n",
        "        print(\"\\nGames in parlay:\")\n",
        "        for game in parlay['games']:\n",
        "            prediction = \"HOME WIN\" if game['predicted_home_win'] else \"AWAY WIN\"\n",
        "            print(f\"  {game['HOME_TEAM']} vs {game['AWAY_TEAM']} - \" \n",
        "                 f\"Prediction: {prediction} (Confidence: {game['home_win_probability']:.4f})\")\n",
        "\n",
        "display_parlays(lr_parlays, \"Logistic Regression\")\n",
        "display_parlays(rf_parlays, \"Random Forest\")"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Simulate ROI"
    ]
    },
    {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Simulate ROI for both models\n",
        "print(\"\\nLogistic Regression ROI Simulation:\")\n",
        "lr_roi = simulate_roi(lr_parlays, lr_results)\n",
        "\n",
        "print(\"\\nRandom Forest ROI Simulation:\")\n",
        "rf_roi = simulate_roi(rf_parlays, rf_results)\n",
        "\n",
        "# Compare ROI visually\n",
        "from src.visualization.visualize import plot_roi\n",
        "\n",
        "# Plot side by side\n",
        "plt.figure(figsize=(15, 6))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "plot_roi(lr_roi)\n",
        "plt.title('Logistic Regression ROI')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plot_roi(rf_roi)\n",
        "plt.title('Random Forest ROI')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Conclusion\n",
        "\n",
        "Based on our analysis, we can draw the following conclusions:\n",
        "\n",
        "1. Model Performance: [Add your observations here based on the results]\n",
        "2. Parlay Strategy: [Add your observations here]\n",
        "3. Next Steps: [Add recommendations for improvements]"
    ]
    }
    ],
    "metadata": {
    "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
    },
    "language_info": {
    "codemirror_mode": {
        "name": "ipython",
        "version": 3
    },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.8.10"
    }
    },
    "nbformat": 4,
    "nbformat_minor": 4
    }
]

# Create notebooks
create_notebook(data_exploration_cells, './01_data_exploration.ipynb')
create_notebook(model_evaluation_cells, './02_model_evaluation.ipynb')