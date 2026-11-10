"""Download LOVW-8.5K from Hugging Face (https://huggingface.co/datasets/Ouroboros/LOVW-8.5K).

By default only the files used in the paper are downloaded (ICA-cleaned 0.3-40 Hz data of the 17 paper
sessions, about 0.5 GB). Use --all for the full dataset (about 4.8 GB, includes the 0.3-3 Hz files,
intermediate processing stages and two sessions not used in the paper).
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.sessions import DEFAULT_DATA_ROOT, HF_REPO, download_patterns  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--all', action='store_true', help='download the whole dataset')
    args = ap.parse_args()
    patterns = None if args.all else download_patterns() + ['README.md', 'croissant.json']
    path = snapshot_download(HF_REPO, repo_type='dataset', local_dir=args.data_root, allow_patterns=patterns)
    print(f'Dataset in {path}')
