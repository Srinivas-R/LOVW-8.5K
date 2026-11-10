"""Within-participant 4-letter classification with chronological 5-fold cross-validation.

Reproduces Table 1 (ME movement-locked, all models) and Figure 3a (EEGNet in the three settings).

Examples:
    python scripts/classify.py --model eegnet --setting me_movement            # all participants
    python scripts/classify.py --model eegnet --setting mi_cue --participants P1
    python scripts/classify.py --model cbramod --mode lp --setting me_movement
    python scripts/classify.py --model reve --mode ft --setting me_movement   # downloads REVE (remote code)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import FM_PARAMS, SPECIALIST_PARAMS, load_trials  # noqa: E402
from lovw.models import CBraModClassifier, EEGNet, REVEClassifier, build_braindecode, cbramod_weights  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT, PARTICIPANTS  # noqa: E402
from lovw.training import cross_validate, set_seed  # noqa: E402

# Defaults per model: (input params, learning rate, epochs)
DEFAULTS = {
    'eegnet': (SPECIALIST_PARAMS, 1e-2, 100),
    'conformer': (SPECIALIST_PARAMS, 1e-3, 100),
    'sparcnet': (SPECIALIST_PARAMS, 1e-3, 100),
    'cbramod': (FM_PARAMS, 1e-3, 30),
    'reve': (FM_PARAMS, 1e-3, 30),
}


def build(model, mode, n_channels, n_times, sfreq, weights=None):
    if model == 'eegnet':
        return EEGNet(n_channels=n_channels, n_times=n_times, sampling_rate=sfreq)
    if model in ('conformer', 'sparcnet'):
        return build_braindecode(model, n_channels, n_times, sfreq)
    if model == 'cbramod':
        return CBraModClassifier(weights, n_channels, n_times, linear_probe=(mode == 'lp'))
    if model == 'reve':
        return REVEClassifier(FM_PARAMS['electrodes'], n_times, linear_probe=(mode == 'lp'))
    raise ValueError(model)


def fold_of_trial(folds, n):
    fold = np.full(n, -1)
    for k, (_, test_idx) in enumerate(folds):
        fold[test_idx] = k
    return fold.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=list(DEFAULTS), default='eegnet')
    ap.add_argument('--mode', choices=['ft', 'lp'], default='ft', help='foundation models: full fine-tuning or linear probe')
    ap.add_argument('--setting', choices=['me_movement', 'me_cue', 'mi_cue'], default='me_movement')
    ap.add_argument('--participants', nargs='+', default=PARTICIPANTS)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/classify')
    args = ap.parse_args()

    params, lr, epochs = DEFAULTS[args.model]
    lr = args.lr or lr
    epochs = args.epochs or epochs
    weights = cbramod_weights() if args.model == 'cbramod' else None
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model + (f'-{args.mode}' if args.model in ('cbramod', 'reve') else '')

    summary = {}
    for participant in args.participants:
        set_seed(args.seed)
        data = load_trials(participant, args.setting, params, args.data_root)
        n, c, t = data.X.shape
        print(f'{tag} {args.setting} {participant}: {n} trials, lr={lr}, epochs={epochs}', flush=True)
        start = time.time()
        module = build(args.model, args.mode, c, t, params['sampling_rate'], weights)
        preds, fold_accs, folds = cross_validate(module, data.X, data.y, data.session_ids, lr, epochs,
                                                 args.batch_size, args.device)
        result = {
            'model': tag, 'setting': args.setting, 'participant': participant, 'seed': args.seed,
            'lr': lr, 'epochs': epochs, 'n_trials': int(n),
            'fold_accs': fold_accs, 'mean_fold_acc': float(np.mean(fold_accs)),
            'std_fold_acc': float(np.std(fold_accs)),
            'trial_acc': float((preds == data.y.numpy()).mean()),
            'y': data.y.tolist(), 'pred': preds.tolist(),
            'session': data.session_ids.tolist(), 'onset_s': np.round(data.onsets_s, 2).tolist(),
            'fold': fold_of_trial(folds, n),
            'minutes': round((time.time() - start) / 60, 1),
        }
        summary[participant] = result['mean_fold_acc']
        name = f'{tag}_{args.setting}_{participant}_seed{args.seed}.json'
        with open(os.path.join(args.out_dir, name), 'w') as f:
            json.dump(result, f)
        print(f'{tag} {args.setting} {participant}: mean fold accuracy {result["mean_fold_acc"]:.3f} '
              f'(± {result["std_fold_acc"]:.3f} across folds)', flush=True)

    accs = np.array(list(summary.values()))
    print(f'{tag} {args.setting}: participant average {accs.mean():.3f} ± {accs.std():.3f} '
          f'({", ".join(f"{p} {a:.3f}" for p, a in summary.items())})')


if __name__ == '__main__':
    main()
