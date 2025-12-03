import pandas as pd

def load_raw_data(path):
    """
    Load the raw XLS dataset.
    This assumes xlrd==1.2.0 is installed to support old .xls files.
    """
    return pd.read_excel(path, header=1)

def convert_xls_to_csv(input_path, output_path):

    df = load_raw_data(input_path)
    df.to_csv(output_path, index=False)
    print(f"[✓] CSV saved to: {output_path}")
    
def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:

    df_prep = df.copy()

    if "ID" in df_prep.columns:
        df_prep = df_prep.drop(columns=["ID"])

    # Convert all columns to numeric
    df_prep = df_prep.apply(pd.to_numeric, errors="coerce")

    # Fill missing values (dataset has none, but safe)
    df_prep = df_prep.fillna(0)

    return df_prep


if __name__ == "__main__":
    input_file = "data/raw/default of credit card clients.xls"
    output_file = "data/processed/credit_default.csv"

    print("[i] Converting XLS → CSV ...")
    convert_xls_to_csv(input_file, output_file)
    print("[✓] Conversion complete.")
