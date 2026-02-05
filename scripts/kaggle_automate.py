import os
import subprocess
import time
import json
import shutil
import zipfile

# --- CONFIG ---
KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME" # <--- UPDATE THIS
KERNEL_SLUG = "gmail-classifier-training"
DATASET_SLUG = "gmail-training-data"
TRAINING_DATA = os.path.join('data', 'training_data.csv')
METADATA_FILE = os.path.join('config', 'kernel-metadata.json')

def run_cmd(cmd):
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    return result.stdout

def automate_kaggle():
    if KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME":
        print("❌ Please edit 'scripts/kaggle_automate.py' and set your KAGGLE_USERNAME first.")
        return

    # 1. Prepare Dataset Folder
    dataset_dir = 'kaggle_dataset'
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    shutil.copy(TRAINING_DATA, os.path.join(dataset_dir, 'training_data.csv'))
    
    # Create dataset metadata if it doesn't exist
    meta_path = os.path.join(dataset_dir, 'dataset-metadata.json')
    if not os.path.exists(meta_path):
        with open(meta_path, 'w') as f:
            json.dump({
                "title": "Gmail Training Data",
                "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}]
            }, f)
        run_cmd(['python', '-m', 'kaggle', 'datasets', 'create', '-p', dataset_dir, '--dir-mode', 'zip'])
    else:
        run_cmd(['python', '-m', 'kaggle', 'datasets', 'version', '-p', dataset_dir, '-m', "Update weekly data", '--dir-mode', 'zip'])

    # 2. Update Kernel Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Error: {METADATA_FILE} not found.")
        return

    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
    meta['id'] = f"{KAGGLE_USERNAME}/{KERNEL_SLUG}"
    meta['dataset_sources'] = [f"{KAGGLE_USERNAME}/{DATASET_SLUG}"]
    
    # Ensure code_file path is correct for the push (it will be pushed from root)
    meta['code_file'] = os.path.join('scripts', 'kaggle_kernel.py')
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. Push Kernel
    print("⏳ Pushing kernel to Kaggle...")
    # We push from root so paths in metadata are relative to root
    run_cmd(['python', '-m', 'kaggle', 'kernels', 'push', '-k', METADATA_FILE, '-p', '.'])

    # 4. Wait for completion
    print("🚦 Polling for completion (this may take 5-10 minutes)...")
    while True:
        status_out = run_cmd(['python', '-m', 'kaggle', 'kernels', 'status', f"{KAGGLE_USERNAME}/{KERNEL_SLUG}"])
        if not status_out: break
        print(status_out.strip())
        if "complete" in status_out.lower():
            break
        if "error" in status_out.lower():
            print("❌ Kernel failed on Kaggle.")
            return
        time.sleep(30)

    # 5. Download output
    print("⬇️ Downloading new model weights...")
    run_cmd(['python', '-m', 'kaggle', 'kernels', 'output', f"{KAGGLE_USERNAME}/{KERNEL_SLUG}", '-p', 'models/'])

    # 6. Unzip and update local model
    zip_path = os.path.join('models', 'model_output.zip')
    if os.path.exists(zip_path):
        print("📦 Unpacking new model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('models/')
        
        # Cleanup
        os.remove(zip_path)
        print("✅ SUCCESS! Local model updated with Kaggle weights.")
    else:
        print(f"❌ Could not find {zip_path}.")

if __name__ == "__main__":
    automate_kaggle()
