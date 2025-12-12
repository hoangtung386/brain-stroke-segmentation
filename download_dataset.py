#!/usr/bin/env python3
"""
Simple dataset downloader for this project.

Usage examples:
  python download_dataset.py                # download default image+mask into data/
  python download_dataset.py --image-id FILEID --mask-id FILEID
  python download_dataset.py --image-name myimg.png --mask-name mymask.png

The script downloads files from Google Drive (handles large-file confirm token) and
saves them into `data/image/` and `data/mask/` by default.
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import subprocess


def ensure_requests():
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
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # fallback: try to find confirm token in the HTML
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)&", response.text)
    if m:
        return m.group(1)
    return None


def save_response_content(response, destination, chunk_size=32768):
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
                    print(f"\rDownloaded {downloaded}/{total} bytes ({pct:.1f}%)", end='', flush=True)
    if total:
        print()


def download_file_from_google_drive(file_id: str, destination: str):
    """Download a file from Google Drive into `destination` path."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    response.raise_for_status()
    save_response_content(response, destination)


def mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)


DEFAULT_IMAGE_ID = '157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk'
DEFAULT_MASK_ID = '1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download dataset files into data/ folder')
    parser.add_argument('--image-id', default=DEFAULT_IMAGE_ID, help='Google Drive file id for the image')
    parser.add_argument('--mask-id', default=DEFAULT_MASK_ID, help='Google Drive file id for the mask')
    parser.add_argument('--image-name', default='image.png', help='Filename to save the image as')
    parser.add_argument('--mask-name', default='mask.png', help='Filename to save the mask as')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--no-overwrite', action='store_true', help='Do not overwrite existing files')
    args = parser.parse_args(argv)

    image_dir = os.path.join(args.data_dir, 'image')
    mask_dir = os.path.join(args.data_dir, 'mask')
    mkdir_p(image_dir)
    mkdir_p(mask_dir)

    image_path = os.path.join(image_dir, args.image_name)
    mask_path = os.path.join(mask_dir, args.mask_name)

    if args.no_overwrite and os.path.exists(image_path):
        print(f"Skipping download; image already exists at {image_path}")
    else:
        print(f"Downloading image to {image_path} ...")
        download_file_from_google_drive(args.image_id, image_path)

    if args.no_overwrite and os.path.exists(mask_path):
        print(f"Skipping download; mask already exists at {mask_path}")
    else:
        print(f"Downloading mask to {mask_path} ...")
        download_file_from_google_drive(args.mask_id, mask_path)

    print('Done.')


if __name__ == '__main__':
    main()
