"""
Automated Dataset Downloader with Resumption
---------------------------------------------
Downloads and verifies the Kaggle Credit Card Fraud Detection dataset
(creditcard.csv) for the AIML-02 Fraud & Drift Monitoring project.
Supports HTTP Range resumption, chunked streaming, and validation.
"""

import os
import sys
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")
PRIMARY_URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
EXPECTED_SIZE = 150828752  # Exact size in bytes

def ensure_dataset(filepath=DATA_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    current_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    if current_size >= EXPECTED_SIZE:
        print(f"[OK] Complete dataset verified at: {filepath} ({current_size / (1024*1024):.2f} MB)")
        return filepath
        
    print(f"Dataset download required. Total target size: {EXPECTED_SIZE / (1024*1024):.1f} MB")
    if current_size > 0:
        print(f"Resuming download from byte {current_size:,} ({current_size / (1024*1024):.1f} MB)...")
        
    max_retries = 10
    retry_count = 0
    
    while current_size < EXPECTED_SIZE and retry_count < max_retries:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            if current_size > 0:
                headers["Range"] = f"bytes={current_size}-"
                
            req = urllib.request.Request(PRIMARY_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as response, open(filepath, 'ab' if current_size > 0 else 'wb') as f:
                block_size = 512 * 1024  # 512KB chunks
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    current_size += len(chunk)
                    pct = min(100.0, (current_size / EXPECTED_SIZE) * 100)
                    sys.stdout.write(f"\r[Downloading] {current_size/(1024*1024):.1f} MB / {EXPECTED_SIZE/(1024*1024):.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
                    
            current_size = os.path.getsize(filepath)
            if current_size >= EXPECTED_SIZE:
                print(f"\n[Success] Dataset download complete! Size: {current_size/(1024*1024):.2f} MB")
                return filepath
        except Exception as e:
            retry_count += 1
            current_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            print(f"\n[Notice] Connection interrupted ({e}). Resuming attempt {retry_count}/{max_retries}...")
            time.sleep(2)
            
    if current_size >= EXPECTED_SIZE:
        return filepath
    else:
        raise RuntimeError(f"Failed to complete download after {max_retries} attempts. Current size: {current_size} bytes.")

if __name__ == "__main__":
    ensure_dataset()
