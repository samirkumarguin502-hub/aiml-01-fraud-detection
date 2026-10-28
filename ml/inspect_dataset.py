from pathlib import Path
import pandas as pd


def inspect_dataset(file_path: str) -> None:
    """Inspect a CSV dataset without modifying the source file."""
    path = Path(file_path)

    if not path.is_file():
        print(f"Dataset not found: {path}")
        print("Place your CSV dataset in the project data folder.")
        return

    df = pd.read_csv(path, nrows=1000)

    print("\n--- Dataset Inspection ---")
    print(f"File: {path.name}")
    print(f"Rows inspected: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    dataset = input("Enter the CSV file path: ").strip().strip('"')
    inspect_dataset(dataset)