"""Temporal-jitter simulation on P1's ME movement-locked trials (Figure 3b and the mixed-effects analysis).

Each trial's 1 s window is shifted by a random offset drawn uniformly from [-J, +J] around movement onset,
simulating unknown onset times. Two conditions:
  * no_augmentation: one fixed random offset per trial (training and test trials alike);
  * augmentation:    training trials get a new random offset every time they are drawn (data
                     augmentation); validation and test trials keep their fixed offset.
EEGNet is trained with chronological 5-fold cross-validation for each jitter half-width J.
Reimplemented for this release from the description in the paper (the original script is not available).
One result file is written per seed, so seeds can run in parallel.

Example:
    python scripts/jitter.py                     # 11 half-widths x 2 conditions x 5 seeds x 5 folds
    python scripts/jitter.py --seeds 42 --jitters 0 0.2 0.4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import SPECIALIST_PARAMS, load_trials  # noqa: E402
from lovw.models import EEGNet  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT  # noqa: E402
from lovw.training import accuracy, chronological_folds, set_seed, train_plain  # noqa: E402


def crop(X_wide, offsets, max_shift, n_times):
    """Cut n_times samples from each wide trial, shifted by `offsets` samples relative to onset."""
    offsets = torch.as_tensor(offsets, device=X_wide.device)
    idx = (max_shift + offsets)[:, None] + torch.arange(n_times, device=X_wide.device)[None, :]
    return torch.gather(X_wide, 2, idx[:, None, :].expand(-1, X_wide.shape[1], -1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--participant', default='P1')
    ap.add_argument('--jitters', type=float, nargs='+', default=[round(0.05 * i, 2) for i in range(11)],
                    help='jitter half-widths in seconds')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 19, 99, 7, 2024])
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/jitter')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = SPECIALIST_PARAMS
    sfreq, n_times = p['sampling_rate'], p['pre_center_samples'] + p['post_center_samples']
    max_shift = int(round(max(args.jitters) * sfreq))
    data = load_trials(args.participant, 'me_movement', p, args.data_root, extra_samples=max_shift)
    X_wide, y = data.X.to(args.device), data.y
    folds = chronological_folds(data.session_ids)
    n, c = len(y), X_wide.shape[1]

    for seed in args.seeds:
        rows = []
        for jitter in args.jitters:
            J = int(round(jitter * sfreq))
            rng = np.random.default_rng([seed, J])
            fixed = rng.integers(-J, J + 1, size=n)                    # one fixed offset per trial
            X_fixed = crop(X_wide, fixed, max_shift, n_times)
            for k, (train_idx, test_idx) in enumerate(folds):
                val = np.zeros(len(train_idx), bool)
                val[rng.choice(len(train_idx), len(train_idx) // 5, replace=False)] = True
                tr, va = train_idx[~val], train_idx[val]
                for condition in ['no_augmentation', 'augmentation']:
                    augment = None
                    if condition == 'augmentation' and J > 0:
                        X_tr_wide = X_wide[tr]

                        def augment(idx, r, X_tr_wide=X_tr_wide, J=J):
                            return crop(X_tr_wide[idx], r.integers(-J, J + 1, size=len(idx)), max_shift, n_times)
                    set_seed(seed + k)
                    model = train_plain(EEGNet(c, n_times, sfreq), X_fixed[tr], y[tr], args.lr, args.epochs,
                                        device=args.device, X_val=X_fixed[va], y_val=y[va], augment=augment,
                                        seed=seed + k)
                    acc = accuracy(model, X_fixed[test_idx], y[test_idx], args.device)
                    rows.append({'jitter_s': jitter, 'condition': condition, 'seed': seed, 'fold': k, 'acc': acc})
            means = {cond: np.mean([r['acc'] for r in rows if r['jitter_s'] == jitter and r['condition'] == cond])
                     for cond in ['no_augmentation', 'augmentation']}
            print(f'seed {seed}, jitter ±{jitter:.2f} s: ' + ', '.join(f'{k} {v:.3f}' for k, v in means.items()),
                  flush=True)
        with open(os.path.join(args.out_dir, f'jitter_{args.participant}_seed{seed}.json'), 'w') as f:
            json.dump(rows, f)


if __name__ == '__main__':
    main()
