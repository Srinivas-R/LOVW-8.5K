"""Leave-one-session-out (LOSO) decoding with EEGNet for P1 and P2 (Table 3 and Appendix Table).

For each session, EEGNet is trained on all other sessions of the same participant and setting and tested on
the held-out session. Accuracy is reported with its binomial standard error.

Example:
    python scripts/loso.py --setting me_movement
    python scripts/loso.py --setting mi_cue
"""
import argparse
import json
import os
import sys

import numpy as np
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import SPECIALIST_PARAMS, load_trials  # noqa: E402
from lovw.models import EEGNet  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT, SESSIONS  # noqa: E402
from lovw.training import make_net, set_seed  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--setting', choices=['me_movement', 'mi_cue'], default='me_movement')
    ap.add_argument('--participants', nargs='+', default=['P1', 'P2'])
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/loso')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paradigm = 'ME' if args.setting == 'me_movement' else 'MI'
    rows = []
    for participant in args.participants:
        data = load_trials(participant, args.setting, SPECIALIST_PARAMS, args.data_root)
        c, t = data.X.shape[1:]
        for s, session in enumerate(SESSIONS[participant][paradigm]):
            set_seed(args.seed)
            train, test = data.session_ids != s, data.session_ids == s
            with tempfile.TemporaryDirectory() as ckpt:
                net = make_net(EEGNet(c, t, SPECIALIST_PARAMS['sampling_rate']), args.lr, args.epochs, 128,
                               args.device, ckpt)
                net.fit(data.X[train], data.y[train])
                acc = float((net.predict(data.X[test]) == data.y[test].numpy()).mean())
            n = int(test.sum())
            se = float(np.sqrt(acc * (1 - acc) / n))
            rows.append({'participant': participant, 'session': session, 'n_trials': n, 'acc': acc, 'se': se})
            print(f'{participant} {session}: {n} trials, accuracy {acc:.3f} ± {se:.3f}', flush=True)
    with open(os.path.join(args.out_dir, f'loso_{args.setting}_seed{args.seed}.json'), 'w') as f:
        json.dump(rows, f, indent=1)
