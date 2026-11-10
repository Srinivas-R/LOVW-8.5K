"""Neural correlates (Figure 5): MRCP at C1 during handwriting imagery, and Grad-CAM salience of EEGNet.

(a) Trials of P1's first MI session, centered on the go cue, at electrode C1 (mne `plot_image`).
(b) EEGNet is trained on P1's MI cue-locked trials with chronological 5-fold cross-validation. For one fold's
    model, the held-out trials of letter L are averaged, and Grad-CAM salience of that average is shown as
    scalp maps in 100 ms bins from -200 to +800 ms around the go cue.

Example:
    python scripts/interpretability.py
"""
import argparse
import os
import sys
import tempfile

import matplotlib
import mne
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lovw.data import ALL_ELECTRODES, SPECIALIST_PARAMS, TABLET_CHANNELS, load_raws, load_trials  # noqa: E402
from lovw.models import EEGNet  # noqa: E402
from lovw.sessions import DEFAULT_DATA_ROOT  # noqa: E402
from lovw.training import chronological_folds, make_net, set_seed  # noqa: E402


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(1)


def mrcp(raw, out):
    raw = raw.copy().drop_channels(TABLET_CHANNELS)
    mne.add_reference_channels(raw, ref_channels='Cz', copy=False)
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    event_id = {f'{letter}_fixation': i + 1 for i, letter in enumerate('LOVW')}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose='ERROR')
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1, tmax=2, preload=True, verbose='ERROR')
    fig = epochs.plot_image(picks=['C1'], vmin=-12, vmax=12, sigma=1, show=False)[0]
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {out}')


def gradcam(data, fold, device, lr, epochs, seed, out):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    set_seed(seed)
    train_idx, test_idx = chronological_folds(data.session_ids)[fold]
    c, t = data.X.shape[1:]
    with tempfile.TemporaryDirectory() as ckpt:
        net = make_net(EEGNet(c, t, SPECIALIST_PARAMS['sampling_rate']), lr, epochs, 128, device, ckpt)
        net.fit(data.X[train_idx], data.y[train_idx])
    model = net.module_.eval()
    letter_L = test_idx[data.y[test_idx].numpy() == 0]
    x = data.X[letter_L].mean(0, keepdim=True).to(device)              # average of held-out L trials
    print(f'fold {fold}: {len(letter_L)} held-out L trials; averaged input classified as '
          f'{"LOVW"[model(x).argmax().item()]}')
    image_model = torch.nn.Sequential(Squeeze(), model)        # grad-cam expects (batch, 1, channels, time)
    cam = GradCAM(model=image_model, target_layers=[model.conv1, model.conv2, model.conv3[0], model.conv3[1]])
    salience = cam(input_tensor=x.unsqueeze(1), targets=[ClassifierOutputTarget(0)]).squeeze()   # (channels, time)
    pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    pos = np.array([pos[ch][:2] for ch in ALL_ELECTRODES])
    fig, axs = plt.subplots(2, 5, figsize=(14, 5))
    for b in range(10):
        ax = axs[b // 5, b % 5]
        ax.set_title(f'[{b * 100 - 200}, {(b + 1) * 100 - 200}] ms', fontsize=15)
        mne.viz.plot_topomap(salience[:, b * 10:(b + 1) * 10].mean(axis=1), pos, vlim=(0.3, 1.5), axes=ax,
                             contours=6, show=False)
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--participant', default='P1')
    ap.add_argument('--fold', type=int, default=0, help='which cross-validation fold model to explain')
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default='results/figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    first_mi = load_raws(args.participant, 'MI', args.data_root)[0]
    mrcp(first_mi, os.path.join(args.out_dir, f'mrcp_C1_{args.participant}_MI.pdf'))
    data = load_trials(args.participant, 'mi_cue', SPECIALIST_PARAMS, args.data_root)
    gradcam(data, args.fold, args.device, args.lr, args.epochs, args.seed,
            os.path.join(args.out_dir, f'gradcam_{args.participant}_MI_L.pdf'))
