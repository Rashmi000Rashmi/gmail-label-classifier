import pandas as pd
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import torch
import os
import shutil
import json

def train_local():
    print("--- 🧠 Starting FAST Delta Training ---")
    dataset_path = os.path.join('data', 'training_data.csv')
    model_dir = os.path.join('models', 'email_classifier_model')
    progress_file = os.path.join('state', 'training_progress.json')
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run collect_data.py first.")
        return

    # 1. Load the dataset
    df = pd.read_csv(dataset_path)
    label_map = {"Application_Confirmation": 0, "Rejected": 1}
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label']) # Ensure valid labels
    
    # 2. Delta Check: Only train on UNSEEN rows
    last_row_trained = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            state = json.load(f)
            last_row_trained = state.get('last_row_trained') or state.get('last_processed_count') or 0

    total_rows = len(df)
    if total_rows <= last_row_trained:
        print(f"No new data since last training run. Skipping.")
        return

    # Extract only the "Delta" (new data)
    new_data = df.iloc[last_row_trained:].copy()
    print(f"Total Rows: {total_rows} | New for Training: {len(new_data)}")

    # 3. Anchor Replay (Optional but recommended)
    # Mix new data with a tiny sample of old data to preserve memory
    if last_row_trained > 0:
        old_data = df.iloc[:last_row_trained]
        anchor_sample = old_data.sample(n=min(len(old_data), 20), random_state=42)
        training_df = pd.concat([new_data, anchor_sample]).sample(frac=1).reset_index(drop=True)
        print(f"Training on {len(new_data)} new items + {len(anchor_sample)} anchors.")
    else:
        training_df = new_data
        print(f"First-time run: Training on {len(training_df)} items.")

    # 4. Tokenize
    tokenizer_path = model_dir if os.path.exists(model_dir) else "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(examples):
        return tokenizer(examples["full_text"], padding="max_length", truncation=True, max_length=512)
    
    train_dataset = Dataset.from_pandas(training_df).map(tokenize_function, batched=True)

    # 5. Load Model (Warm-start if exists)
    if os.path.exists(model_dir):
        print(f"Warm-starting from {model_dir}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    else:
        print("Cold-starting from DistilBERT base...")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 6. Training Arguments (Lightweight for fast updates)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=4 if device == "cpu" else 8,
        num_train_epochs=3 if len(training_df) < 50 else 2,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        report_to="none"
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 8. Train
    print("Updating weights...")
    trainer.train()

    # 9. Save
    print(f"Saving model to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    with open(progress_file, 'w') as f:
        json.dump({'last_row_trained': total_rows}, f)
        
    print("\n✅ Training complete.")

if __name__ == '__main__':
    train_local()
