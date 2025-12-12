#!/usr/bin/env python3
"""
Dataset downloader for Brain Stroke Segmentation project.
Downloads ZIP files from Google Drive, extracts them, and organizes the data.

Usage examples:
  python download_dataset.py                # download default image+mask zips into data/
  python download_dataset.py --image-id FILEID --mask-id FILEID
  python download_dataset.py --keep-zip     # keep zip files after extraction
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import subprocess
import zipfile
from pathlib import Path


def ensure_requests():
    """Ensure requests package is installed"""
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        print("`requests` package not found — attempting to install it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"]) 
        import requests  # type: ignore
        return requests


requests = ensure_requests()


def get_confirm_token(response):
    """Extract download confirmation token from Google Drive response"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # fallback: try to find confirm token in the HTML
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)&", response.text)
    if m:
        return m.group(1)
    return None


def save_response_content(response, destination, chunk_size=32768):
    """Save response content to file with progress bar"""
    total = response.headers.get('Content-Length')
    try:
        total = int(total) if total is not None else None
    except Exception:
        total = None

    downloaded = 0
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(f"\r  Downloaded {mb_downloaded:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", 
                          end='', flush=True)
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded {mb_downloaded:.1f} MB", end='', flush=True)
    print()  # New line after progress


def download_file_from_google_drive(file_id: str, destination: str):
    """Download a file from Google Drive into `destination` path."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"  Requesting file from Google Drive (ID: {file_id})...")
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        print(f"  Large file detected, confirming download...")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    response.raise_for_status()
    save_response_content(response, destination)


def extract_zip(zip_path: str, extract_to: str):
    """Extract ZIP file to specified directory"""
    print(f"  Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        
        print(f"  Found {total_files} files in archive")
        
        # Extract with progress
        for idx, file in enumerate(file_list, 1):
            zip_ref.extract(file, extract_to)
            if idx % 100 == 0 or idx == total_files:
                pct = idx / total_files * 100
                print(f"\r  Extracted {idx}/{total_files} files ({pct:.1f}%)", 
                      end='', flush=True)
        print()  # New line after progress
    
    print(f"  Extraction complete!")


def count_files(directory: str, extension: str = '.png') -> int:
    """Count files with specific extension in directory and subdirectories"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(extension))
    return count


def mkdir_p(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


# Default Google Drive file IDs (these are ZIP files)
DEFAULT_IMAGE_ID = '157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk'
DEFAULT_MASK_ID = '1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download and extract brain stroke dataset from Google Drive'
    )
    parser.add_argument(
        '--image-id', 
        default=DEFAULT_IMAGE_ID, 
        help='Google Drive file id for the image ZIP'
    )
    parser.add_argument(
        '--mask-id', 
        default=DEFAULT_MASK_ID, 
        help='Google Drive file id for the mask ZIP'
    )
    parser.add_argument(
        '--data-dir', 
        default='data', 
        help='Base data directory'
    )
    parser.add_argument(
        '--keep-zip', 
        action='store_true', 
        help='Keep ZIP files after extraction'
    )
    parser.add_argument(
        '--no-overwrite', 
        action='store_true', 
        help='Skip download if data already exists'
    )
    args = parser.parse_args(argv)

    # Create data directory structure
    data_dir = Path(args.data_dir)
    image_dir = data_dir / 'image'
    mask_dir = data_dir / 'mask'
    
    mkdir_p(str(data_dir))
    mkdir_p(str(image_dir))
    mkdir_p(str(mask_dir))

    # Define temporary zip file paths
    image_zip = data_dir / 'image.zip'
    mask_zip = data_dir / 'mask.zip'

    print("="*60)
    print("Brain Stroke Dataset Downloader")
    print("="*60)

    # Check if data already exists
    if args.no_overwrite:
        image_count = count_files(str(image_dir))
        mask_count = count_files(str(mask_dir))
        
        if image_count > 0 and mask_count > 0:
            print(f"\nData already exists:")
            print(f"  - Images: {image_count} files")
            print(f"  - Masks: {mask_count} files")
            print("\nSkipping download (use without --no-overwrite to re-download)")
            return

    # Download and extract images
    print(f"\n[1/2] Processing Images")
    print("-" * 30)
    
    if image_zip.exists() and not args.no_overwrite:
        print(f"  Removing existing {image_zip}...")
        image_zip.unlink()
    
    if not image_zip.exists():
        print(f"  Downloading image ZIP to {image_zip}...")
        try:
            download_file_from_google_drive(args.image_id, str(image_zip))
            print(f"  Download complete!")
        except Exception as e:
            print(f"\n  Error downloading images: {e}")
            return
    else:
        print(f"  Using existing {image_zip}")

    # Extract images
    try:
        extract_zip(str(image_zip), str(data_dir))
        
        # Count extracted files
        image_count = count_files(str(image_dir))
        print(f"  Total images: {image_count} files")
    except Exception as e:
        print(f"\n  Error extracting images: {e}")
        return

    # Download and extract masks
    print(f"\n[2/2] Processing Masks")
    print("-" * 60)
    
    if mask_zip.exists() and not args.no_overwrite:
        print(f"  Removing existing {mask_zip}...")
        mask_zip.unlink()
    
    if not mask_zip.exists():
        print(f"  Downloading mask ZIP to {mask_zip}...")
        try:
            download_file_from_google_drive(args.mask_id, str(mask_zip))
            print(f"  Download complete!")
        except Exception as e:
            print(f"\n  Error downloading masks: {e}")
            return
    else:
        print(f"  Using existing {mask_zip}")

    # Extract masks
    try:
        extract_zip(str(mask_zip), str(data_dir))
        
        # Count extracted files
        mask_count = count_files(str(mask_dir))
        print(f"  Total masks: {mask_count} files")
    except Exception as e:
        print(f"\n  Error extracting masks: {e}")
        return

    # Clean up ZIP files
    if not args.keep_zip:
        print(f"\n[Cleanup]")
        print("-" * 30)
        if image_zip.exists():
            print(f"  Removing {image_zip}...")
            image_zip.unlink()
        if mask_zip.exists():
            print(f"  Removing {mask_zip}...")
            mask_zip.unlink()
        print(f"  Cleanup complete!")

    # Display summary
    print("\n" + "-"*30)
    print("Dataset Download Complete!")
    print("-"*30)
    print(f"\nData summary:")
    print(f"  - Images: {image_count} files in {image_dir}")
    print(f"  - Masks:  {mask_count} files in {mask_dir}")
    
    # Check data structure
    print(f"\nData structure:")
    image_subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    mask_subdirs = [d for d in mask_dir.iterdir() if d.is_dir()]
    
    if image_subdirs:
        print(f"  - Image subfolders: {len(image_subdirs)}")
        print(f"    Examples: {', '.join([d.name for d in image_subdirs[:3]])}")
    
    if mask_subdirs:
        print(f"  - Mask subfolders: {len(mask_subdirs)}")
        print(f"    Examples: {', '.join([d.name for d in mask_subdirs[:3]])}")


if __name__ == '__main__':
    main()
