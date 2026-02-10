"""
Auto-download utility for datasets not natively supported by torchvision.

Handles downloading + extracting archives from stable URLs (Zenodo, Caltech, etc.).
"""

import os
import tarfile
import urllib.request
import zipfile
from typing import Optional


def download_and_extract(url: str, root: str, filename: Optional[str] = None,
                         extract_root: Optional[str] = None, remove_archive: bool = False) -> str:
    """
    Download a file from URL and extract it.

    Args:
        url: Direct download URL.
        root: Directory to save the downloaded file.
        filename: Override filename (default: inferred from URL).
        extract_root: Directory to extract to (default: same as root).
        remove_archive: Whether to delete the archive after extraction.

    Returns:
        Path to the extracted directory.
    """
    os.makedirs(root, exist_ok=True)

    if filename is None:
        filename = url.split('/')[-1].split('?')[0]

    filepath = os.path.join(root, filename)
    if extract_root is None:
        extract_root = root

    # Download if not already present
    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url}...")
        _download_with_progress(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Archive already exists: {filepath}")

    # Extract
    _extract_archive(filepath, extract_root)

    if remove_archive and os.path.exists(filepath):
        os.remove(filepath)

    return extract_root


def _download_with_progress(url: str, filepath: str):
    """Download with a simple progress indicator."""
    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size if total_size > 0 else 0
        print(f"\r  Progress: {pct}%", end='', flush=True)

    urllib.request.urlretrieve(url, filepath, reporthook=_progress)
    print()  # newline after progress


def _extract_archive(filepath: str, extract_root: str):
    """Extract tar/zip archives."""
    if filepath.endswith('.zip'):
        print(f"Extracting {filepath}...")
        with zipfile.ZipFile(filepath, 'r') as z:
            z.extractall(extract_root)
    elif filepath.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
        print(f"Extracting {filepath}...")
        mode = 'r:gz' if filepath.endswith(('.tar.gz', '.tgz')) else \
               'r:bz2' if filepath.endswith('.tar.bz2') else 'r:'
        with tarfile.open(filepath, mode) as t:
            t.extractall(extract_root)
    else:
        print(f"Skipping extraction (unknown format): {filepath}")
