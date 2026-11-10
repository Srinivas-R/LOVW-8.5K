"""Paired trial-level permutation test between two models (Appendix: Permutation test).

H0: the two models are equally likely to be correct on any given trial. The statistic is the difference in
participant-average accuracy, T = mean_s acc_s(A) - mean_s acc_s(B). In each permutation the two models'
predictions are swapped per trial with probability 1/2; p = fraction of permutations with T_i >= T_obs.
Trials are paired by session and event time, so models with different input windows can be compared.

Example (after running scripts/classify.py for both models):
    python scripts/permutation_test.py results/classify/eegnet_me_movement_P*_seed42.json \
        --vs results/classify/cbramod-ft_me_movement_P*_seed42.json
"""
import argparse
import glob
import json

import numpy as np


def load(paths):
    out = {}
    for p in sorted(set(q for pattern in paths for q in glob.glob(pattern))):
        r = json.load(open(p))
        key = {(s, round(o, 1)): (y, pr) for s, o, y, pr in zip(r['session'], r['onset_s'], r['y'], r['pred'])}
        out[r['participant']] = key
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', nargs='+', help='result files of model A (one per participant)')
    ap.add_argument('--vs', nargs='+', required=True, help='result files of model B')
    ap.add_argument('--n-perm', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    A, B = load(args.a), load(args.vs)
    participants = sorted(set(A) & set(B))
    correct_a, correct_b = [], []
    for p in participants:
        common = sorted(set(A[p]) & set(B[p]))
        correct_a.append(np.array([A[p][k][0] == A[p][k][1] for k in common]))
        correct_b.append(np.array([B[p][k][0] == B[p][k][1] for k in common]))
        print(f'{p}: {len(common)} paired trials, accuracy A {correct_a[-1].mean():.3f}, B {correct_b[-1].mean():.3f}')
    t_obs = np.mean([a.mean() - b.mean() for a, b in zip(correct_a, correct_b)])
    rng = np.random.default_rng(args.seed)
    t_null = np.zeros(args.n_perm)
    for a, b in zip(correct_a, correct_b):
        d = (a.astype(float) - b.astype(float))
        for start in range(0, args.n_perm, 10_000):
            stop = min(start + 10_000, args.n_perm)
            signs = rng.integers(0, 2, size=(stop - start, len(d))) * 2 - 1   # swap A and B per trial
            t_null[start:stop] += (signs * d).mean(axis=1) / len(participants)
    p = float((t_null >= t_obs).mean())
    print(f'T_obs = {t_obs:.4f} (participant-average accuracy A - B), one-sided p = {p:.5f} '
          f'({args.n_perm} permutations; p = 0 means < {1 / args.n_perm:g})')


if __name__ == '__main__':
    main()
