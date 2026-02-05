import subprocess
import os
import sys

def run_step(command):
    print(f"\n🚀 Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error in step: {' '.join(command)}")
        sys.exit(1)

def main():
    print("="*50)
    print("      📧 GMAIL AUTONOMOUS CLASSIFIER 📧")
    print("="*50)
    print("\n[1] Daily Routine: Just Classify Unread Emails")
    print("[2] Update Model: Sync Manual Labels + Fine-tune (Local CPU)")
    print("[3] Update Model: Sync Manual Labels + Fine-tune (Kaggle GPU)")
    print("[4] Advanced: Just Sync Data (No training)")
    print("[Q] Quit")
    
    choice = input("\nSelect an option: ").strip().lower()

    if choice == '1':
        print("\nPhase: Classifying Unread Emails...")
        run_step(['python', 'scripts/classify_emails.py'])
        print("\nPhase: Updating Dashboard Metrics...")
        run_step(['python', 'scripts/extract_metrics.py'])
        
    elif choice == '2':
        print("\nPhase 1: Syncing Labels...")
        run_step(['python', 'scripts/collect_data.py'])
        print("\nPhase 2: Fine-tuning (Local CPU)...")
        run_step(['python', 'scripts/local_train.py'])
        print("\nPhase 3: Classifying Unread Emails...")
        run_step(['python', 'scripts/classify_emails.py'])
        print("\nPhase 4: Updating Dashboard Metrics...")
        run_step(['python', 'scripts/extract_metrics.py'])

    elif choice == '3':
        print("\nPhase 1: Syncing Labels...")
        run_step(['python', 'scripts/collect_data.py'])
        print("\nPhase 2: Fine-tuning (Kaggle GPU)...")
        run_step(['python', 'scripts/kaggle_automate.py'])
        print("\nPhase 3: Classifying Unread Emails...")
        run_step(['python', 'scripts/classify_emails.py'])
        print("\nPhase 4: Updating Dashboard Metrics...")
        run_step(['python', 'scripts/extract_metrics.py'])
        
    elif choice == '4':
        print("\nPhase: Syncing Labels and Preparing Dataset...")
        run_step(['python', 'scripts/collect_data.py'])

    elif choice == 'q':
        sys.exit(0)
    else:
        print("Invalid choice.")

    print("\n" + "="*50)
    print("✅ Completed! Your Gmail is now organized.")
    print("="*50)

if __name__ == "__main__":
    main()
