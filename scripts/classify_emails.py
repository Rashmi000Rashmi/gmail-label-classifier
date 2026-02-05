import os
import torch
import base64
import re
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from bs4 import BeautifulSoup

# --- Configuration ---
MODEL_PATH = os.path.join('models', 'email_classifier_model')
REMOTE_MODEL_ID = "your-username/gmail-classifier" # <-- Friends can set your repo ID here
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CONFIDENCE_THRESHOLD = 0.85
LABEL_MAP = {0: "Application_Confirmation", 1: "Rejected"}
DRY_RUN = False  # Set to False to actually apply labels and mark as read

# Search query: finds unread emails from the last 7 days
# This is more efficient than scanning all time, but flexible enough for daily runs.
from datetime import datetime, timedelta
seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y/%m/%d")
GMAIL_QUERY = f'is:unread after:{seven_days_ago}' 

# Load Model and Tokenizer
try:
    print("Loading model...")
    # 1. Check if local model exists, else try remote
    if os.path.exists(MODEL_PATH):
        load_source = MODEL_PATH
        print(f"✅ Using local model: {MODEL_PATH}")
    else:
        load_source = REMOTE_MODEL_ID
        print(f"🚀 Local model not found. Downloading from Hub: {REMOTE_MODEL_ID}")
    
    # Enable attentions for explainability
    model = DistilBertForSequenceClassification.from_pretrained(load_source, output_attentions=True)
    tokenizer = DistilBertTokenizer.from_pretrained(load_source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✨ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    if load_source == REMOTE_MODEL_ID:
        print("\n💡 TIP: If you are a new user, make sure the REMOTE_MODEL_ID is set correctly in this script.")
    exit(1)

def get_gmail_service():
    token_path = os.path.join('auth', 'token.json')
    if not os.path.exists(token_path):
        print(f"Error: {token_path} not found. Run scripts/check_labels.py first.")
        return None
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    return build('gmail', 'v1', credentials=creds)

def get_key_phrases(text, top_n=3):
    """Identifies words that the model paid most attention to."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Average attention across all heads in the LAST layer
        # Output attentions size: (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions[-1][0] 
        avg_attention = attentions.mean(dim=0) # Average over heads
        
        # We look at the attention from the [CLS] token (index 0) to all other tokens
        cls_attention = avg_attention[0]
        
    # Map tokens back to words and scores
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_importance = []
    
    for i, token in enumerate(tokens):
        # Ignore special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]']: continue
        word_importance.append((token, cls_attention[i].item()))
        
    # Sort by importance
    word_importance.sort(key=lambda x: x[1], reverse=True)
    return [w[0] for w in word_importance[:top_n]]

def clean_email_text(text):
    if not text: return ""
    if BeautifulSoup and ("<html" in text.lower() or "<div" in text.lower()):
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    
    # Identify key phrases for explanation
    key_phrases = get_key_phrases(text)
    
    return predicted_class.item(), confidence.item(), key_phrases

def get_unread_emails(service):
    # Fetch unread emails using the configured query
    results = service.users().messages().list(userId='me', q=GMAIL_QUERY).execute()
    return results.get('messages', [])

def apply_label(service, msg_id, label_name):
    # 1. Get label ID
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    
    label_id = next((l['id'] for l in labels if l['name'] == label_name), None)
    
    if not label_id:
        print(f"Label '{label_name}' not found. Creating it...")
        label_body = {'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
        created_label = service.users().labels().create(userId='me', body=label_body).execute()
        label_id = created_label['id']

    # 2. Add label and remove UNREAD
    service.users().messages().batchModify(
        userId='me',
        body={
            'ids': [msg_id],
            'addLabelIds': [label_id],
            'removeLabelIds': ['UNREAD']
        }
    ).execute()

def ensure_labels_exist(service):
    """Checks for required labels and creates them if missing."""
    print("📋 Checking Gmail labels...")
    required_labels = ["Application_Confirmation", "Rejected", "Uncertain"]
    
    results = service.users().labels().list(userId='me').execute()
    existing_labels = [l['name'] for l in results.get('labels', [])]
    
    for label_name in required_labels:
        if label_name not in existing_labels:
            print(f"➕ Creating label: {label_name}")
            label_body = {
                'name': label_name, 
                'labelListVisibility': 'labelShow', 
                'messageListVisibility': 'show'
            }
            service.users().labels().create(userId='me', body=label_body).execute()

def main():
    service = get_gmail_service()
    if not service: return

    # Ensure environment is ready for friends
    ensure_labels_exist(service)

    messages = get_unread_emails(service)
    if not messages:
        print("No new unread emails found.")
        return

    print(f"Found {len(messages)} unread emails. Processing...")
    
    report = {
        "Application_Confirmation": [],
        "Rejected": [],
        "Uncertain": []
    }

    for m in messages:
        try:
            msg = service.users().messages().get(userId='me', id=m['id']).execute()
            
            # Extract Subject and Body
            headers = msg['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
            
            # Extract content
            snippet = msg.get('snippet', '')
            full_text = clean_email_text(subject + " " + snippet)

            # Predict
            pred_idx, conf, key_phrases = predict(full_text)
            
            if conf >= CONFIDENCE_THRESHOLD:
                label = LABEL_MAP[pred_idx]
                print(f"[{label}] '{subject}' (Conf: {conf:.2f})")
                if not DRY_RUN:
                    apply_label(service, m['id'], label)
                report[label].append(subject)
            else:
                print(f"[UNCERTAIN] '{subject}' (Conf: {conf:.2f}) - Applying 'Uncertain' label.")
                if not DRY_RUN:
                    apply_label(service, m['id'], "Uncertain")
                report["Uncertain"].append(subject)

        except Exception as e:
            print(f"Error processing message {m['id']}: {e}")

    print("\n" + "="*40)
    print("         FINAL CLASSIFICATION REPORT")
    print("="*40)
    
    for category, subjects in report.items():
        print(f"\n📌 {category.upper()} ({len(subjects)})")
        if subjects:
            for s in subjects[:10]: # Show first 10
                print(f"  - {s}")
            if len(subjects) > 10:
                print(f"  ... and {len(subjects)-10} more")
        else:
            print("  (None)")
    
    if DRY_RUN:
        print("\n[!] NOTE: This was a DRY RUN. No labels were actually applied in Gmail.")


if __name__ == '__main__':
    main()
