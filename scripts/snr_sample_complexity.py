"""Training-set size vs. test-time SNR (Figure 4).

EEGNet is trained on a random fraction of P1's ME movement-locked trials, excluding the final 160 trials,
which form a fixed test set. Test-time SNR is increased by averaging k test trials of the same letter
(k = 1, 2, 4, 8) before classification; training always uses single trials.

Example:
    python scripts/snr_sample_complexity.py
    python scripts/snr_sample_complexity.py --seeds 42 --train-ratios 1.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import SPECIALIST_PARAMS, average_same_letter, load_trials  # noqa: E402
from lovw.models import EEGNet  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT  # noqa: E402
from lovw.training import predict, set_seed, train_plain  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--participant', default='P1')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 19, 99])
    ap.add_argument('--avg-over', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--train-ratios', type=float, nargs='+', default=[round(0.1 * i, 1) for i in range(1, 11)])
    ap.add_argument('--n-test', type=int, default=160)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--depth', type=int, default=2, help='EEGNet spatial filters per temporal filter (D)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/snr')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = SPECIALIST_PARAMS
    data = load_trials(args.participant, 'me_movement', p, args.data_root)
    X, y = data.X, data.y
    n, c, t = X.shape
    test_idx = np.arange(n - args.n_test, n)            # the last trials of the final session
    pool = np.arange(n - args.n_test)

    rows = []
    for seed in args.seeds:
        for ratio in args.train_ratios:
            set_seed(seed)
            rng = np.random.default_rng(seed)
            train_idx = np.sort(rng.choice(pool, int(ratio * len(pool)), replace=False))
            model = train_plain(EEGNet(c, t, p['sampling_rate'], D=args.depth), X[train_idx], y[train_idx],
                                args.lr, args.epochs, args.batch_size, args.device, seed=seed)
            for k in args.avg_over:
                Xa, ya = average_same_letter(X[test_idx].numpy(), y[test_idx].numpy(), k)
                acc = float((predict(model, torch.from_numpy(Xa), args.device) == ya).mean())
                rows.append({'seed': seed, 'train_ratio': ratio, 'train_size': len(train_idx), 'avg_over': k,
                             'n_test': len(ya), 'acc': acc})
            print(f'seed {seed}, {len(train_idx)} training trials: ' +
                  ', '.join(f'avg{r["avg_over"]} {r["acc"]:.3f}' for r in rows[-len(args.avg_over):]), flush=True)
    with open(os.path.join(args.out_dir, f'snr_{args.participant}.json'), 'w') as f:
        json.dump(rows, f)


if __name__ == '__main__':
    main()
