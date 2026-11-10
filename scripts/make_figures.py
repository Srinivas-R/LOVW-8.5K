"""Summary tables and figures from the result files written by the other scripts.

  * Table 1 / Figure 3a: results/classify/*.json  (scripts/classify.py)
  * Figure 3b + mixed-effects analysis: results/jitter/jitter_P1_seed*.json  (scripts/jitter.py)
  * Figure 4: results/snr/snr_P1.json  (scripts/snr_sample_complexity.py)
  * Reaction-time figure (Appendix): computed from the data

Example:
    python scripts/make_figures.py
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import SPECIALIST_PARAMS, epoch_session, load_raws  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT  # noqa: E402

SETTING_NAMES = {'me_movement': 'ME movement-locked', 'me_cue': 'ME cue-locked', 'mi_cue': 'MI cue-locked'}


def classification(results_dir, out_dir):
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, '*.json')):
        r = json.load(open(path))
        runs[(r['model'], r['setting'], r['participant'])].append(r)
    if not runs:
        return
    rows = []
    for (model, setting, participant), rs in sorted(runs.items()):
        rows.append({'model': model, 'setting': setting, 'participant': participant, 'n_seeds': len(rs),
                     'acc': np.mean([r['mean_fold_acc'] for r in rs]),
                     'fold_std': np.mean([r['std_fold_acc'] for r in rs])})
    df = pd.DataFrame(rows)
    print('\nPer-participant accuracy (mean over seeds of the mean over 5 chronological folds):')
    print(df.pivot_table(index=['model', 'setting'], columns='participant', values='acc').round(3).to_string())
    summary = df.groupby(['model', 'setting'])['acc'].agg(['mean', 'std', 'count'])
    summary['std'] = df.groupby(['model', 'setting'])['acc'].std(ddof=0)
    print('\nParticipant average ± SD across participants (Table 1 / Figure 3a):')
    print(summary.round(3).to_string())

    eeg = df[df.model == 'eegnet']
    if len(eeg):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        participants = sorted(eeg.participant.unique())
        width = 0.25
        for i, setting in enumerate(SETTING_NAMES):
            sub = eeg[eeg.setting == setting].set_index('participant').reindex(participants)
            x = np.arange(len(participants)) + (i - 1) * width
            ax.bar(x, sub.acc, width, yerr=sub.fold_std, capsize=2, label=SETTING_NAMES[setting])
        ax.axhline(0.25, color='black', linestyle='--', linewidth=2, label='Chance performance')
        ax.set_xticks(np.arange(len(participants)), participants)
        ax.set_ylabel('Accuracy')
        ax.legend()
        fig.savefig(os.path.join(out_dir, 'fig3a_settings.pdf'), bbox_inches='tight')


def jitter(pattern, out_dir):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return
    df = pd.DataFrame([row for p in paths for row in json.load(open(p))])
    fig, ax = plt.subplots(figsize=(5, 4))
    for condition, label in [('augmentation', 'Data augmentation'), ('no_augmentation', 'No data augmentation')]:
        sub = df[df.condition == condition]
        per_fold = sub.groupby(['jitter_s', 'fold'])['acc'].mean().reset_index()   # average seeds per fold
        g = per_fold.groupby('jitter_s')['acc']
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(ddof=0), capsize=3, marker='o', label=label)
    ax.axhline(0.25, color='black', linestyle='--', label='Chance')
    ax.set_xlabel('Jitter half-width (s)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'fig3b_jitter.pdf'), bbox_inches='tight')
    print('\nJitter (mean accuracy over seeds and folds):')
    print(df.pivot_table(index='jitter_s', columns='condition', values='acc').round(3).to_string())

    import statsmodels.formula.api as smf
    df['aug'] = (df.condition == 'augmentation').astype(int)
    df['seed_fold'] = df.seed.astype(str) + '_' + df.fold.astype(str)
    model = smf.mixedlm('acc ~ aug * jitter_s', df, groups=df['seed'],
                        vc_formula={'fold': '0 + C(fold)'}).fit()
    print('\nMixed-effects model acc ~ condition x jitter, random intercepts for seed and fold:')
    print(model.summary().tables[1].loc[['aug', 'aug:jitter_s']].to_string())


def snr(path, out_dir):
    if not os.path.exists(path):
        return
    df = pd.DataFrame(json.load(open(path)))
    summary = df.groupby(['train_size', 'avg_over'])['acc'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, sub in summary.groupby('avg_over'):
        ax.errorbar(sub.train_size, sub['mean'], yerr=sub['std'], capsize=4, marker='o',
                    label=f'Avg over {k} trials')
    ax.axhline(0.25, color='black', linestyle='--', label='Chance acc.')
    ax.set_xlabel('Train set size')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'fig4_snr_sample_complexity.pdf'), bbox_inches='tight')
    print('\nTest-time SNR vs training-set size (mean ± SD over seeds):')
    print(summary.pivot(index='train_size', columns='avg_over', values='mean').round(3).to_string())


def reaction_times(data_root, out_dir, session_index=2):
    raw = load_raws('P1', 'ME', data_root)[session_index]
    _, _, cue = epoch_session(raw, SPECIALIST_PARAMS, 'fixation')
    _, _, pen = epoch_session(raw, SPECIALIST_PARAMS, 'pen_down')
    rt = np.array([pen[pen > c].min() - c for c in cue if (pen > c).any()]) / SPECIALIST_PARAMS['sampling_rate']
    rt = rt[rt < 1.5]                                   # pen-down within the trial
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist((rt - rt.mean()) * 1000, bins=40)
    ax.set_xlabel('Mean-subtracted reaction time (ms)')
    ax.set_ylabel('Trials')
    fig.savefig(os.path.join(out_dir, 'appendix_reaction_times.pdf'), bbox_inches='tight')
    print(f'\nReaction time (go cue to pen down), P1 session {session_index + 1}: '
          f'mean {rt.mean() * 1000:.0f} ms, SD {rt.std() * 1000:.0f} ms, n = {len(rt)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    classification(os.path.join(args.results_dir, 'classify'), args.out_dir)
    jitter(os.path.join(args.results_dir, 'jitter', 'jitter_P1_seed*.json'), args.out_dir)
    snr(os.path.join(args.results_dir, 'snr', 'snr_P1.json'), args.out_dir)
    reaction_times(args.data_root, args.out_dir)
