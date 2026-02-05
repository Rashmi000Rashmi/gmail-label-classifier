# 📧 Gmail Label Classifier

As a new grad student when I was looking for the first internship/job I use to apply aroung 30+ jobs each day, therefore I used to get application submit confirmation emails, rejection emails, interview call emails, etc. It was a repatitive task to manually label them and I recently see my friend going through the same phase. So I decided to create a system that can automatically label the emails and provide a live dashboard to track the progress. An autonomous, AI-powered system to organize your job search emails. This project uses NLP (DistilBERT) to automatically label emails as **Application_Confirmation**, **Rejected**, or **Uncertain**, providing a live dashboard to track your progress.

For the data collection part, I have majority of emails for Application submit confirmations and rejections, so the model is biased towards these two labels. I highly recommend to check the uncertain emails and label them manually to improve the model's performance. I have also added a feature to retrain the model with the new labels. You can retrain the model by running the `auto_run.py` script and it will retrain the model with the new labels.

---

## 🚀 Key Features

- **Autonomous Classification**: automatically scans unread emails and applies Gmail labels.
- **Incremental Intelligence**: Smart memory ensures it never downloads or trains on the same email twice.
- **Privacy-Safe Dashboard**: Live Streamlit/Plotly dashboard showing your application trends without exposing private data.
- **Cloud-Ready**: Native support for **Kaggle GPU** training and **Hugging Face Hub** model hosting.
- **Friend Edition**: Simplified workflow for others to use your "trained brain" with zero setup.

---

## 📁 Project Structure

```text
├── auth/            # 🔒 Gmail API tokens and credentials.json
├── config/          # Kaggle kernel configuration
├── data/            # Local data archives (raw_emails.csv, metrics.csv)
├── models/          # Local AI model weights
├── scripts/         # Core logic (sync, train, classify, visualize)
├── state/           # 🧠 Sync and training memory files
├── auto_run.py      # Main entry point for the "Teacher" (You)
├── friend_run.py    # Simplified entry point for "Users" (Friends)
└── requirements.txt # Project dependencies
```

---

## 🛠️ Setup & Usage

### 1. Prerequisites
- Python 3.11+
- Google Cloud Project with Gmail API enabled (Download `credentials.json` into `auth/`).

### 2. Installations
```bash
pip install -r requirements.txt
```

### 3. Running the Project
- **For You (Full Control)**:
  Run `python auto_run.py` to sync data, train the model (Local/Kaggle), and classify emails.
  
- **For Friends (Quick Labeling)**:
  Run `python friend_run.py`. It will download the pre-trained model and start organizing Gmail immediately.

---

## 📊 Live Visualization
Your job search metrics are extracted into a privacy-safe `data/metrics.csv` (counts only).
- **View Live**: [https://gmail-label-classifier.streamlit.app/](https://gmail-label-classifier.streamlit.app/)
- **Local Preview**: `streamlit run scripts/app.py`

---

## 🤝 Model Sharing
The model is hosted on Hugging Face: **[Rashmi000/Gmail_Label](https://huggingface.co/Rashmi000/Gmail_Label)**.
New users don't need to train; the code will automatically fetch this "trained brain" from the Hub.

---

## 🛡️ Privacy & Security
- **Strict Gitignore**: Your private emails (`data/*.csv`) and API tokens (`auth/`) are **never** committed to GitHub.
- **Metrics Only**: The dashboard only receives date-based counts, ensuring your subject lines and bodies remain local.
