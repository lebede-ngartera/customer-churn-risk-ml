"""
Download the IBM Telco Customer Churn dataset.

Data Source: IBM Sample Data Sets
URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

This script downloads the raw data to data/raw/ directory.
The data should be treated as immutable after download.
"""

import os
from pathlib import Path

import requests


def download_telco_data(output_dir: str = "data/raw") -> str:
    """
    Download the Telco Customer Churn dataset.
    
    Args:
        output_dir: Directory to save the downloaded file.
        
    Returns:
        Path to the downloaded file.
        
    Raises:
        requests.RequestException: If download fails.
    """
    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / "telco_customer_churn.csv"
    
    # Check if file already exists
    if file_path.exists():
        print(f"File already exists: {file_path}")
        return str(file_path)
    
    print(f"Downloading from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(file_path, "wb") as f:
        f.write(response.content)
    
    print(f"Downloaded successfully: {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024:.1f} KB")
    
    return str(file_path)


if __name__ == "__main__":
    # When run as script, download to project root's data/raw
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw"
    download_telco_data(str(output_dir))
