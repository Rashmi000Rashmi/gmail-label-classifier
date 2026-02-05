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
    print("      📧 GMAIL CLASSIFIER (Friend's Edition) 📧")
    print("="*50)
    print("\nThis script will set up your Gmail labels and organize your emails.")
    print("\n--- 🛠️ Setup ---")
    
    # 1. Check for credentials
    auth_dir = 'auth'
    if not os.path.exists(os.path.join(auth_dir, 'credentials.json')):
        print(f"❌ Error: 'auth/credentials.json' missing.")
        print("Please place your Google Cloud 'credentials.json' in the 'auth' folder first.")
        sys.exit(1)

    # 2. Authenticate and check labels
    print("\nStep 1: Authenticating with Gmail...")
    run_step(['python', 'scripts/check_labels.py'])

    # 3. Running Classifier
    print("\nStep 2: Classifying Unread Emails...")
    print("This will download the AI model from the cloud if this is your first run.")
    run_step(['python', 'scripts/classify_emails.py'])

    print("\n" + "="*50)
    print("✅ Done! Your Gmail labels are now organized.")
    print("Labels created: Application_Confirmation, Rejected, Uncertain")
    print("="*50)

if __name__ == "__main__":
    main()
