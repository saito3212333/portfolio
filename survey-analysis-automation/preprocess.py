import pandas as pd
import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    """Load settings from a YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def apply_cleaning(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean the dataframe based on rules defined in config.
    Prints a traceability log for each step.
    """
    rules = config['cleaning_rules']
    initial_count = len(df)

    print(f"{'Data Cleaning Trace':^40}")
    print("-" * 40)
    print(f"{'Initial records':<25}: {initial_count:>6}")

    # Step 1: Filter by completion time
    df = df[df['completion_time_sec'] >= rules['min_completion_time_sec']]
    count_after_time = len(df)
    print(f"{'Time filter applied':<25}: {count_after_time:>6} (Dropped: {initial_count - count_after_time})")

    # Step 2: Drop rows with missing target values
    df = df.dropna(subset=[rules['target_column']])
    count_after_null = len(df)
    print(f"{'Null values removed':<25}: {count_after_null:>6} (Dropped: {count_after_time - count_after_null})")

    # Step 3: Filter by age range
    min_age, max_age = rules['age_range']
    df = df[df['age'].between(min_age, max_age)]
    final_count = len(df)
    print(f"{'Age range filter applied':<25}: {final_count:>6} (Dropped: {count_after_null - final_count})")
    print("-" * 40)

    return df

def optimize_memory(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert columns to optimized types to save RAM."""
    types = config['data_types']

    # Convert to categorical for low-cardinality strings
    for col in types['categories']:
        df[col] = df[col].astype('category')

    # Downcast integers (e.g., int64 to int8)
    for col in types['integers']:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    return df