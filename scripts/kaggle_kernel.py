import pandas as pd
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import torch
import os
import zipfile

# 1. Setup paths
# When we push with a dataset, Kaggle puts it in /kaggle/input
DATASET_NAME = 'gmail-training-data'
DATA_FILE = f'/kaggle/input/{DATASET_NAME}/training_data.csv'
OUTPUT_DIR = './email_classifier_model'

print(f"--- 🚀 Kaggle Training Started ---")

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found in Kaggle environment.")
    # List files to help debug
    print("Files in current dir:", os.listdir('.'))
    exit(1)

# 2. Load and prepare data
df = pd.read_csv(DATA_FILE)
label_map = {"Application_Confirmation": 0, "Rejected": 1}
df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])

print(f"Training on {len(df)} emails...")

dataset = Dataset.from_pandas(df)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["full_text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

# 3. Load Model
# Note: On Kaggle, we usually start fresh or load from a Kaggle Dataset.
# For simplicity, we start fresh here. 
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16, # Higher batch size on GPU
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no", # We save manually at the end
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
)

# 6. Train
trainer.train()

# 7. Save Model
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 8. Zip for easy download via API
with zipfile.ZipFile('model_output.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            zipf.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), os.path.join(OUTPUT_DIR, '..')))

print("--- ✅ Kaggle Training Complete! ---")
