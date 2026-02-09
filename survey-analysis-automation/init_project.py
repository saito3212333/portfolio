from pathlib import Path

def initialize_project():
    """Create the necessary directory structure for the analysis project."""

    # Define a list of directories to create
    directories = [
        Path("data/raw"),
        Path("data/processed"),
        Path("reports/figures"),
        Path("src"), # If you want to put preprocess.py here
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created: {directory}")
        else:
            print(f"Already exists: {directory}")

if __name__ == "__main__":
    initialize_project()