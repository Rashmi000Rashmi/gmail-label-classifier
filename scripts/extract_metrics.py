import pandas as pd
import os
import json

def extract_metrics():
    raw_file = os.path.join('data', 'raw_emails.csv')
    metrics_file = os.path.join('data', 'metrics.csv')
    
    if not os.path.exists(raw_file):
        print(f"❌ Error: {raw_file} not found. Run a sync first.")
        return

    print("--- 🛡️ Extracting Privacy-Safe Metrics ---")
    
    try:
        # 1. Load raw data
        df = pd.read_csv(raw_file)
        
        # 2. Convert date to standard YYYY-MM-DD
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['date_only'] = df['date_dt'].dt.date
        
        # 3. Drop rows with invalid dates
        df = df.dropna(subset=['date_only'])
        
        # 4. Group by Date and Label to get counts
        # We ONLY keep Date, Label, and Count
        metrics_df = df.groupby(['date_only', 'label']).size().reset_index(name='count')
        
        # 5. Save to the safe file
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Safe metrics saved to: {metrics_file}")
        print(f"💡 This file contains ONLY counts and dates. No private email content.")

    except Exception as e:
        print(f"❌ Failed to extract metrics: {e}")

if __name__ == "__main__":
    extract_metrics()
