import os
from huggingface_hub import HfApi, create_repo
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def push_to_hub():
    model_path = os.path.join('models', 'email_classifier_model')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Local model not found at {model_path}")
        return

    print("--- 🚀 Hugging Face Hub Upload ---")
    repo_id = input("Enter your Hugging Face repo name (e.g., 'your-username/gmail-classifier'): ").strip()
    hf_token = input("Enter your Hugging Face Write Token: ").strip()

    if not repo_id or not hf_token:
        print("❌ Error: Repo ID and Token are required.")
        return

    try:
        print(f"⏳ Loading model from {model_path}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)

        print(f"🆕 Creating repository '{repo_id}' (if it doesn't exist)...")
        try:
            create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)
        except Exception as e:
            print(f"⚠️ Repo creation warning (it might already exist): {e}")

        print("📤 Pushing model to Hub...")
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)

        print(f"\n✅ SUCCESS! Your model is now public at: https://huggingface.co/{repo_id}")
        print("\n💡 Now your friends can use your model by setting their repo_id in scripts/classify_emails.py")

    except Exception as e:
        print(f"❌ Failed to push to hub: {e}")

if __name__ == "__main__":
    push_to_hub()
