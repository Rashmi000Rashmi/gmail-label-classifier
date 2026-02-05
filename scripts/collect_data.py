import os
import csv
import base64
import re
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Please run: pip install beautifulsoup4")
    BeautifulSoup = None

# --- CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
DATA_DIR = 'data'
RAW_FILE = os.path.join(DATA_DIR, 'raw_emails.csv')
TRAINING_FILE = os.path.join(DATA_DIR, 'training_data.csv')
STATE_FILE = os.path.join('state', 'sync_state.json')

def final_clean(text):
    """Deep cleaning of email text for training."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    noise_phrases = [
        r'Please do not reply to this email\.?',
        r'This is an unattended mailbox\.?',
        r'Replies will not be read\.?',
        r'You can find more information here',
        r'Follow us on .*',
        r'Visit our Newsroom .*'
    ]
    for phrase in noise_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_email_text(text):
    """Basic cleaning to remove HTML and noise."""
    if not text:
        return ""
    if BeautifulSoup and ("<html" in text.lower() or "<div" in text.lower()):
        try:
            soup = BeautifulSoup(text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=' ')
        except Exception:
            pass
            
    quote_patterns = [r'On\s+.*\s+wrote:.*', r'-+\s*Original Message\s*-+.*', r'From:\s+.*', r'Sent:\s+.*', r'>.*']
    for pattern in quote_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_full_text(msg):
    """Extracts text content from Gmail message payload."""
    payload = msg.get('payload', {})
    def decode_data(data):
        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    def walk_parts(parts):
        t_cont, h_cont = "", ""
        for part in parts:
            mime = part.get('mimeType')
            body = part.get('body', {})
            data = body.get('data')
            if mime == 'text/plain' and data:
                t_cont += decode_data(data)
            elif mime == 'text/html' and data:
                h_cont += decode_data(data)
            elif 'parts' in part:
                t, h = walk_parts(part['parts'])
                t_cont += t
                h_cont += h
        return t_cont, h_cont

    parts = payload.get('parts', [])
    if parts:
        plain, html = walk_parts(parts)
        content = plain if plain else html
    else:
        data = payload.get('body', {}).get('data')
        content = decode_data(data) if data else ""

    if not content:
        content = msg.get('snippet', '')
    return clean_email_text(content)

def get_data():
    if not os.path.exists(os.path.join('auth', 'token.json')):
        print("Please run scripts/check_labels.py first!")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Load last sync timestamp
    last_sync_ts = 0  
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            last_sync_ts = json.load(f).get('last_sync_ts', 0)
    
    new_sync_ts = int(datetime.now().timestamp() * 1000)
    creds = Credentials.from_authorized_user_file(os.path.join('auth', 'token.json'), SCOPES)
    service = build('gmail', 'v1', credentials=creds)

    target_labels = ['Application_Confirmation', 'Rejected']
    all_emails = []

    for label_name in target_labels:
        try:
            if last_sync_ts == 0:
                print(f"Syncing ALL historical emails for: {label_name}...")
                query = f'label:"{label_name}"'
            else:
                query_date = datetime.fromtimestamp(last_sync_ts / 1000).strftime("%Y/%m/%d")
                print(f"Syncing: {label_name} since {query_date}...")
                query = f'label:"{label_name}" after:{query_date}'

            page_token = None
            while True:
                results = service.users().messages().list(userId='me', q=query, pageToken=page_token).execute()
                messages = results.get('messages', [])

                for m in messages:
                    try:
                        msg = service.users().messages().get(userId='me', id=m['id']).execute()
                        msg_ts = int(msg.get('internalDate', 0))
                        
                        if msg_ts <= last_sync_ts:
                            continue

                        headers = msg['payload'].get('headers', [])
                        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
                        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown")
                        date_val = next((h['value'] for h in headers if h['name'].lower() == 'date'), "Unknown")
                        
                        content = get_full_text(msg)
                        all_emails.append({
                            'date': date_val,
                            'sender': sender,
                            'label': label_name,
                            'subject': subject,
                            'text': content
                        })
                    except Exception as e:
                        print(f"Error on message {m['id']}: {e}")
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
        except Exception as e:
            print(f"Error fetching label {label_name}: {e}")

    if all_emails:
        # 1. Save RAW DATA
        df_raw = pd.DataFrame(all_emails)
        raw_exists = os.path.exists(RAW_FILE)
        df_raw.to_csv(RAW_FILE, mode='a', index=False, header=not raw_exists)
        
        # 2. Save TRAINING DATA (Processed)
        df_train = df_raw.copy()
        df_train['full_text'] = df_train['subject'].fillna('') + " " + df_train['text'].fillna('')
        df_train['full_text'] = df_train['full_text'].apply(final_clean)
        df_train = df_train[df_train['full_text'].str.len() > 30]
        df_train = df_train[['label', 'full_text']]
        
        train_exists = os.path.exists(TRAINING_FILE)
        df_train.to_csv(TRAINING_FILE, mode='a', index=False, header=not train_exists)
        
        with open(STATE_FILE, 'w') as f:
            json.dump({'last_sync_ts': new_sync_ts}, f)
            
        print(f"Success! Collected and processed {len(all_emails)} emails.")
    else:
        print("No new emails found.")

if __name__ == '__main__':
    get_data()
