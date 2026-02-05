import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_metrics():
    data_file = os.path.join('data', 'raw_emails.csv')
    
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found. Run a sync first.")
        return

    print("--- 📊 Generating Job Search Insights ---")
    
    try:
        df = pd.read_csv(data_file)
        
        # 1. Basic Stats
        counts = df['label'].value_counts()
        print("\nSummary Counts:")
        print(counts)

        # 2. Daily Trends
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df_daily = df.groupby([df['date_dt'].dt.date, 'label']).size().unstack(fill_value=0)

        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Distribution
        plt.subplot(1, 2, 1)
        counts.plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
        plt.title('Email Distribution')
        plt.ylabel('')

        # Subplot 2: Trends
        plt.subplot(1, 2, 2)
        df_daily.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('Daily Trends')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Category')

        plt.tight_layout()
        
        output_path = os.path.join('data', 'job_search_metrics.png')
        plt.savefig(output_path)
        print(f"\n✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate visualization: {e}")

if __name__ == "__main__":
    visualize_metrics()
 village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village village
