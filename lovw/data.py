"""Loading LOVW-8.5K sessions and cutting them into letter trials."""
import mne
import numpy as np
import torch
from torch.utils.data import Dataset

from .sessions import DEFAULT_DATA_ROOT, SESSIONS, session_file

# Electrode order used throughout the paper. Cz (the online reference) is not stored in the files and
# is added back as an all-zero channel before common average re-referencing.
ALL_ELECTRODES = ['Fp1', 'Fz', 'F3', 'FC5', 'FC3', 'FC1', 'FCz', 'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3',
                  'CP1', 'Oz', 'Fp2', 'F4', 'FC6', 'FC4', 'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4',
                  'CP2', 'Pz', 'CPz', 'Cz']
LETTERS = ['L', 'O', 'V', 'W']
TABLET_CHANNELS = ['Vx', 'Vy', 'penDown']

# Specialist models (EEGNet, Conformer, SPaRCNet): 100 Hz, 1 s window [-200 ms, +800 ms] around the event.
SPECIALIST_PARAMS = {
    "sampling_rate": 100,
    "pre_center_samples": 20,
    "post_center_samples": 80,
    "scale_to_100_microvolts": False,   # data in microvolts
    "electrodes": ALL_ELECTRODES,
}
# Foundation models (CBraMod, REVE): 200 Hz, 4 s window [-1 s, +3 s] around the event, in units of 100 uV.
FM_PARAMS = {
    "sampling_rate": 200,
    "pre_center_samples": 200,
    "post_center_samples": 600,
    "scale_to_100_microvolts": True,
    "electrodes": ALL_ELECTRODES,
}

# Paper settings: which paradigm's sessions to load and which event to center the trials on.
SETTINGS = {
    "me_movement": ("ME", "pen_down"),   # ME movement-locked (true movement onset)
    "me_cue": ("ME", "fixation"),        # ME cue-locked (go cue)
    "mi_cue": ("MI", "fixation"),        # MI cue-locked (go cue)
}


def load_raws(participant, paradigm, data_root=DEFAULT_DATA_ROOT, sessions=None):
    """Read the sessions of one participant and paradigm ('ME' or 'MI'), in chronological order."""
    sessions = sessions or SESSIONS[participant][paradigm]
    return [mne.io.read_raw_eeglab(session_file(s, data_root), preload=True, verbose='ERROR') for s in sessions]


def epoch_session(raw, params, event='pen_down', extra_samples=0):
    """Cut one session into letter trials.

    Returns X (trials x channels x time), y (letter index), onset sample of each trial, in chronological order.
    `extra_samples` widens the window by that many samples on each side (used for temporal jitter).
    """
    raw = raw.copy().resample(sfreq=params['sampling_rate'], verbose='ERROR')
    data = raw.get_data()
    eeg = data[:-len(TABLET_CHANNELS)]
    ch2idx = {ch: i for i, ch in enumerate(raw.ch_names[:-len(TABLET_CHANNELS)])}
    if 'Cz' not in ch2idx:
        ch2idx['Cz'] = eeg.shape[0]
        eeg = np.concatenate([eeg, np.zeros((1, eeg.shape[1]))], axis=0)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)                    # common average reference
    eeg = eeg[[ch2idx[ch] for ch in params['electrodes']]]

    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    onsets, labels = [], []
    for label, letter in enumerate(LETTERS):
        sel = events[events[:, 2] == event_id[f'{letter}_{event}'], 0]
        onsets.append(sel)
        labels.append(np.full(len(sel), label))
    onsets, labels = np.concatenate(onsets), np.concatenate(labels)

    pre = params['pre_center_samples'] + extra_samples
    post = params['post_center_samples'] + extra_samples
    keep = (onsets - pre >= 0) & (onsets + post <= eeg.shape[1])   # drop trials running off the recording
    onsets, labels = onsets[keep], labels[keep]
    order = np.argsort(onsets)                                     # chronological order
    onsets, labels = onsets[order], labels[order]
    X = np.stack([eeg[:, o - pre:o + post] for o in onsets])
    X = X * (1e4 if params['scale_to_100_microvolts'] else 1e6)   # volts -> 100 uV or uV
    return X.astype(np.float32), labels.astype(np.int64), onsets


class LetterTrials(Dataset):
    """All trials of a list of sessions, concatenated in the given (chronological) order.

    Attributes: X (N x C x T float tensor), y (N,), session_ids (N,): index of the session of each trial,
    onsets_s (N,): time of the centering event within its session, in seconds.
    """

    def __init__(self, raws, params, event='pen_down', extra_samples=0):
        Xs, ys, sids, onsets = [], [], [], []
        for i, raw in enumerate(raws):
            X, y, onset = epoch_session(raw, params, event, extra_samples)
            Xs.append(X), ys.append(y), sids.append(np.full(len(y), i))
            onsets.append(onset / params['sampling_rate'])
        self.params = params
        self.X = torch.from_numpy(np.concatenate(Xs))
        self.y = torch.from_numpy(np.concatenate(ys))
        self.session_ids = np.concatenate(sids)
        self.onsets_s = np.concatenate(onsets)       # event time of each trial within its session (s)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_trials(participant, setting, params=SPECIALIST_PARAMS, data_root=DEFAULT_DATA_ROOT, extra_samples=0):
    """Trials of one participant in one paper setting ('me_movement', 'me_cue' or 'mi_cue')."""
    paradigm, event = SETTINGS[setting]
    return LetterTrials(load_raws(participant, paradigm, data_root), params, event, extra_samples)


def average_same_letter(X, y, k, rng=None):
    """Average groups of k test trials of the same letter (test-time SNR boost).

    Trials are grouped in order within each letter; incomplete groups are dropped.
    """
    if k == 1:
        return X, y
    Xa, ya = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if rng is not None:
            idx = rng.permutation(idx)
        for g in range(len(idx) // k):
            Xa.append(X[idx[g * k:(g + 1) * k]].mean(axis=0))
            ya.append(label)
    return np.stack(Xa), np.array(ya)
