import os

# List of directories that need __init__.py files
dirs = [
    'src',
    'src/data',
    'src/features',
    'src/models',
    'src/visualization'
]

# Create __init__.py files
for dir_path in dirs:
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Create __init__.py file
    init_file = os.path.join(dir_path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Initialize package\n')
        print(f"Created {init_file}")

print("All __init__.py files created")