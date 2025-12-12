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


def ensure_gdown():
    """Ensure gdown package is installed"""
    try:
        import gdown
        return gdown
    except Exception:
        print("`gdown` package not found — installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"]) 
        import gdown
        return gdown


def extract_zip(zip_path: str, extract_to: str):
    """Extract ZIP file to specified directory"""
    print(f"  Extracting {zip_path}...")
    
    # Verify it's a valid ZIP file
    if not zipfile.is_zipfile(zip_path):
        print(f"  ✗ Error: {zip_path} is not a valid ZIP file")
        print(f"  The file might be corrupted or download failed.")
        
        # Check file size
        file_size = os.path.getsize(zip_path)
        print(f"  File size: {file_size / 1024:.1f} KB")
        
        if file_size < 1024 * 100:  # Less than 100KB
            print(f"  File is too small - likely an error page from Google Drive")
            # Show first 500 chars to debug
            with open(zip_path, 'r', errors='ignore') as f:
                content = f.read(500)
                if '<html' in content.lower() or 'google' in content.lower():
                    print(f"  Detected HTML content - download failed")
        
        raise Exception(f"Invalid ZIP file: {zip_path}")
    
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
    
    print(f"  ✓ Extraction complete!")


def count_files(directory: str, extension: str = '.png') -> int:
    """Count files with specific extension in directory and subdirectories"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(extension))
    return count


def mkdir_p(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def download_from_gdrive(file_id: str, output_path: str, quiet: bool = False):
    """Download file from Google Drive using gdown"""
    gdown = ensure_gdown()
    
    url = f'https://drive.google.com/uc?id={file_id}'
    
    print(f"  Downloading from Google Drive...")
    print(f"  File ID: {file_id}")
    
    try:
        gdown.download(url, output_path, quiet=quiet, fuzzy=True)
        
        # Verify download
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  ✓ Downloaded {file_size:.1f} MB")
            return True
        else:
            print(f"  ✗ Download failed - file not found")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


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
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress gdown progress output'
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
            print(f"\n✓ Data already exists:")
            print(f"  - Images: {image_count} files")
            print(f"  - Masks: {mask_count} files")
            print("\nSkipping download (use without --no-overwrite to re-download)")
            return

    # Download and extract images
    print(f"\n[1/2] Processing Images")
    print("-" * 60)
    
    # Remove old zip if exists
    if image_zip.exists():
        print(f"  Removing existing {image_zip.name}...")
        image_zip.unlink()
    
    # Download
    success = download_from_gdrive(args.image_id, str(image_zip), quiet=args.quiet)
    
    if not success or not image_zip.exists():
        print(f"\n✗ Failed to download images")
        print(f"\nTroubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. Verify the Google Drive file ID: {args.image_id}")
        print(f"  3. Make sure the file is publicly accessible")
        print(f"  4. Try using gdown directly: gdown {args.image_id}")
        return

    # Extract images
    try:
        extract_zip(str(image_zip), str(data_dir))
        
        # Count extracted files
        image_count = count_files(str(image_dir))
        print(f"  ✓ Total images: {image_count} files")
        
        if image_count == 0:
            print(f"  ⚠ Warning: No image files found after extraction")
            
    except Exception as e:
        print(f"\n✗ Error extracting images: {e}")
        print(f"\nThe downloaded file might not be a valid ZIP.")
        print(f"Please check the file manually: {image_zip}")
        return

    # Download and extract masks
    print(f"\n[2/2] Processing Masks")
    print("-" * 60)
    
    # Remove old zip if exists
    if mask_zip.exists():
        print(f"  Removing existing {mask_zip.name}...")
        mask_zip.unlink()
    
    # Download
    success = download_from_gdrive(args.mask_id, str(mask_zip), quiet=args.quiet)
    
    if not success or not mask_zip.exists():
        print(f"\n✗ Failed to download masks")
        print(f"\nTroubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. Verify the Google Drive file ID: {args.mask_id}")
        print(f"  3. Make sure the file is publicly accessible")
        print(f"  4. Try using gdown directly: gdown {args.mask_id}")
        return

    # Extract masks
    try:
        extract_zip(str(mask_zip), str(data_dir))
        
        # Count extracted files
        mask_count = count_files(str(mask_dir))
        print(f"  ✓ Total masks: {mask_count} files")
        
        if mask_count == 0:
            print(f"  ⚠ Warning: No mask files found after extraction")
            
    except Exception as e:
        print(f"\n✗ Error extracting masks: {e}")
        print(f"\nThe downloaded file might not be a valid ZIP.")
        print(f"Please check the file manually: {mask_zip}")
        return

    # Clean up ZIP files
    if not args.keep_zip:
        print(f"\n[Cleanup]")
        print("-" * 60)
        if image_zip.exists():
            print(f"  Removing {image_zip.name}...")
            image_zip.unlink()
        if mask_zip.exists():
            print(f"  Removing {mask_zip.name}...")
            mask_zip.unlink()
        print(f"  ✓ Cleanup complete!")

    # Display summary
    print("\n" + "="*60)
    print("✓ Dataset Download Complete!")
    print("="*60)
    print(f"\nData summary:")
    print(f"  - Images: {image_count} files in {image_dir}")
    print(f"  - Masks:  {mask_count} files in {mask_dir}")
    
    # Check data structure
    print(f"\nData structure:")
    image_subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    mask_subdirs = [d for d in mask_dir.iterdir() if d.is_dir()]
    
    if image_subdirs:
        print(f"  - Image subfolders: {len(image_subdirs)}")
        print(f"    Examples: {', '.join([d.name for d in list(image_subdirs)[:3]])}")
    
    if mask_subdirs:
        print(f"  - Mask subfolders: {len(mask_subdirs)}")
        print(f"    Examples: {', '.join([d.name for d in list(mask_subdirs)[:3]])}")
    
    if image_count > 0 and mask_count > 0:
        print(f"\n✓ Ready to train!")
        print(f"  Next step: python train.py")
    else:
        print(f"\n⚠ Warning: Some data might be missing")
        print(f"  Please verify the downloaded files")
    
    print("="*60)


if __name__ == '__main__':
    main()
