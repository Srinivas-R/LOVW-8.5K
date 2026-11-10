"""Naive cross-subject transfer (Appendix: Cross-subject transfer).

EEGNet is trained on all ME movement-locked trials of one participant and tested on each other participant.

Example:
    python scripts/cross_subject.py
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import SPECIALIST_PARAMS, load_trials  # noqa: E402
from lovw.models import EEGNet  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT, PARTICIPANTS  # noqa: E402
from lovw.training import make_net, set_seed  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--setting', default='me_movement', choices=['me_movement', 'me_cue', 'mi_cue'])
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/cross_subject')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = {p: load_trials(p, args.setting, SPECIALIST_PARAMS, args.data_root) for p in PARTICIPANTS}
    rows = []
    for train_p in PARTICIPANTS:
        set_seed(args.seed)
        X, y = data[train_p].X, data[train_p].y
        with tempfile.TemporaryDirectory() as ckpt:
            net = make_net(EEGNet(X.shape[1], X.shape[2], SPECIALIST_PARAMS['sampling_rate']), args.lr,
                           args.epochs, 128, args.device, ckpt)
            net.fit(X, y)
            for test_p in PARTICIPANTS:
                if test_p == train_p:
                    continue
                acc = float((net.predict(data[test_p].X) == data[test_p].y.numpy()).mean())
                n = len(data[test_p].y)
                rows.append({'train': train_p, 'test': test_p, 'n_test': n, 'acc': acc,
                             'se': float(np.sqrt(acc * (1 - acc) / n))})
                print(f'train {train_p} -> test {test_p}: accuracy {acc:.3f} ± {rows[-1]["se"]:.3f} ({n} trials)',
                      flush=True)
    with open(os.path.join(args.out_dir, f'cross_subject_{args.setting}_seed{args.seed}.json'), 'w') as f:
        json.dump(rows, f, indent=1)
