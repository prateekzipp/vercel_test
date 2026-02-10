import os
import time
import requests
import json
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Details taken from your screenshot:
RESOURCE_NAME = "labellingcompliance"
ANALYZER_ID = "dabur_demo" 
API_VERSION = "2025-11-01"

# ⚠️ PASTE YOUR KEY HERE
# Find this in Azure Portal -> Your Resource -> "Keys and Endpoint"
API_KEY = ""

# The document you want to analyze (local file path, PDF)
DOCUMENT_PATH = "DABUR_ALL_PAGES copy.pdf"

# Full Endpoint URL
ENDPOINT = f"https://{RESOURCE_NAME}.cognitiveservices.azure.com"
ANALYZE_BINARY_URL = f"{ENDPOINT}/contentunderstanding/analyzers/{ANALYZER_ID}:analyzeBinary?api-version={API_VERSION}"

headers = {
    "Ocp-Apim-Subscription-Key": API_KEY,
    "Content-Type": "application/octet-stream"
}

# ==========================================
# 2. THE "CLIENT" LOGIC (Mimicking the Sample)
# ==========================================

def analyze_document():
    print(f"🚀 Submitting document (PDF) to analyzer: {ANALYZER_ID}...")
    
    # Check if file exists
    file_path = Path(DOCUMENT_PATH)
    if not file_path.exists() or not file_path.is_file():
        print(f"❌ Error: PDF file not found at {DOCUMENT_PATH}")
        return None
    
    print(f"✅ PDF file found: {DOCUMENT_PATH}")
    
    # Read the binary file data
    try:
        with open(DOCUMENT_PATH, "rb") as file:
            file_bytes = file.read()
        print(f"📤 Uploading {len(file_bytes)} bytes...")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Step A: POST Request with binary data (begin_analyze_binary)
    try:
        response = requests.post(ANALYZE_BINARY_URL, headers=headers, data=file_bytes)
        response.raise_for_status() # Raise error for bad status codes
        
        # Get the Result ID to track the job
        result_id = response.json()["id"]
        print(f"✅ Submission successful! Tracking ID: {result_id}")
        return result_id
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error submitting request: {e}")
        if response is not None:
             print(response.text)
        return None

def poll_for_results(result_id):
    # Step B: Polling Loop (poll_result)
    result_url = f"{ENDPOINT}/contentunderstanding/analyzerResults/{result_id}?api-version={API_VERSION}"
    print("⏳ Waiting for analysis to complete...")
    
    while True:
        time.sleep(2) # Wait 2 seconds between checks
        
        response = requests.get(result_url, headers=headers)
        if response.status_code != 200:
            print(f"Error polling: {response.status_code}")
            break
            
        data = response.json()
        status = data.get("status")
        
        if status == "Succeeded":
            print("\n🎉 Analysis Complete!")
            return data
        elif status in ["Failed", "Canceled"]:
            print("❌ Analysis Failed.")
            print(data)
            return None
        else:
            print(f"   Status: {status}...")

# ==========================================
# 3. RUNNING THE CODE
# ==========================================

# 1. Start Analysis
job_id = analyze_document()

if job_id:
    # 2. Wait for Result
    final_result = poll_for_results(job_id)
    
    # 3. Print the Output (extracted fields only: value, bounding boxes, and other data)
    if final_result:
        contents = final_result["result"]["contents"]
        
        print("\n" + "="*80)
        print("EXTRACTED FIELDS")
        print("="*80)
        
        for idx, item in enumerate(contents):
            print(f"\n📦 Content Item {idx + 1}")
            print("-" * 80)
            
            fields = item.get("fields", {})
            
            if not fields:
                print("No fields found in this content item.")
                continue
            
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    print(f"\n🔹 {field_name}: {field_data}")
                    continue
                # Value (may be missing or empty)
                value_str = field_data.get("value") if field_data.get("value") is not None else field_data.get("valueString")
                if value_str is None or (isinstance(value_str, str) and value_str.strip() == ""):
                    value_str = "(no value)"
                print(f"\n🔹 {field_name}")
                print(f"   value: {value_str}")
                # Bounding box(es) and any other keys
                for key, value in field_data.items():
                    if key in ("value", "valueString"):
                        continue
                    if isinstance(value, dict):
                        print(f"   {key}: {value}")
                    elif isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            for i, elem in enumerate(value):
                                print(f"   {key}[{i}]: {elem}")
                        else:
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
        print("\n" + "="*80)