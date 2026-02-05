import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def main():
    creds = None
    token_path = os.path.join('auth', 'token.json')
    cred_path = os.path.join('auth', 'credentials.json')

    # token.json stores the user's access and refresh tokens
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(cred_path):
                print(f"Error: {cred_path} not found. Please place it in the auth/ folder.")
                return
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Ensure auth directory exists
        if not os.path.exists('auth'):
            os.makedirs('auth')
            
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    print("\n--- Your Gmail Labels ---")
    for label in labels:
        print(f"- {label['name']} (ID: {label['id']})")

if __name__ == '__main__':
    main()