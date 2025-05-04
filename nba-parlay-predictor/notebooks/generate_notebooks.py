import nbformat as nbf
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Function to create a notebook
def create_notebook(filename, title, description):
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add a markdown cell with title and description
    markdown_cell = nbf.v4.new_markdown_cell(
        f"# {title}\n\n{description}"
    )
    
    # Add a code cell for imports
    imports_cell = nbf.v4.new_code_cell(
        "# Import necessary libraries\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Set plot style\n"
        "sns.set_style('whitegrid')\n"
        "plt.rcParams['figure.figsize'] = (12, 8)"
    )
    
    # Add a code cell for setting up path
    path_cell = nbf.v4.new_code_cell(
        "# Add the src directory to the path so we can import our modules\n"
        "import sys\n"
        "import os\n\n"
        "# Get the absolute path to the project root\n"
        "project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
        "if project_root not in sys.path:\n"
        "    sys.path.append(project_root)\n"
        "print(f\"Added {project_root} to Python path\")"
    )
    
    # Add cells to the notebook
    nb.cells = [markdown_cell, imports_cell, path_cell]
    
    # Save the notebook
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Created {filename}")

# Create the data exploration notebook
create_notebook(
    './01_data_exploration.ipynb',
    'NBA Parlay Betting Analysis - Data Exploration',
    'This notebook explores the NBA game data and betting odds to understand patterns that might be useful for parlay betting.'
)

# Create the model evaluation notebook
create_notebook(
    './02_model_evaluation.ipynb',
    'NBA Parlay Betting Analysis - Model Evaluation',
    'This notebook evaluates different models for NBA game predictions and parlay generation.'
)

print("Both notebooks created successfully!")