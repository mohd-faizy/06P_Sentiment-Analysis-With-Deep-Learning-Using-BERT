#!/usr/bin/env python3
"""
Utility script to verify and download the SMILE Twitter Emotion dataset.
Dataset Source: Wang et al. (2016) / SMILE Project
URL: https://raw.githubusercontent.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/master/smile-annotations-final.csv
"""

import os
import urllib.request

DATA_DIR = 'data'
DATA_FILENAME = 'smile-annotations-final.csv'
TARGET_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

# Primary & fallback download URLs
URLS = [
    'https://raw.githubusercontent.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/master/smile-annotations-final.csv',
    'https://raw.githubusercontent.com/apoudel1021/SENTIMENT_ANALYSIS_USING_BERT_PYTORCH/master/smile-annotations-final.csv',
    'https://figshare.com/ndownloader/files/4988956'
]

def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if already present in data/ or root
    if os.path.exists(TARGET_PATH):
        size_kb = os.path.getsize(TARGET_PATH) / 1024
        print(f"[OK] Dataset already exists at '{TARGET_PATH}' ({size_kb:.1f} KB).")
        return TARGET_PATH
    
    if os.path.exists(DATA_FILENAME):
        size_kb = os.path.getsize(DATA_FILENAME) / 1024
        print(f"[OK] Dataset found in root directory '{DATA_FILENAME}' ({size_kb:.1f} KB).")
        # Copy to data/
        with open(DATA_FILENAME, 'rb') as src, open(TARGET_PATH, 'wb') as dst:
            dst.write(src.read())
        print(f"     Copied to '{TARGET_PATH}'.")
        return TARGET_PATH
    
    print("[*] Downloading SMILE Twitter Emotion dataset...")
    for url in URLS:
        try:
            print(f"    Fetching from: {url}")
            urllib.request.urlretrieve(url, TARGET_PATH)
            if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) > 10000:
                size_kb = os.path.getsize(TARGET_PATH) / 1024
                print(f"[SUCCESS] Downloaded '{TARGET_PATH}' successfully ({size_kb:.1f} KB).")
                return TARGET_PATH
        except Exception as e:
            print(f"    Failed ({e}), trying next source...")
            
    raise RuntimeError("Failed to download dataset from all mirror sources.")

if __name__ == '__main__':
    ensure_dataset()
