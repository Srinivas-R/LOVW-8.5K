"""Sessions of the LOVW-8.5K dataset (https://huggingface.co/datasets/Ouroboros/LOVW-8.5K).

Folder names follow the dataset: <participant>_S<session number>_<ME|MI>, numbered as in the
data-collection table in the paper's appendix. Lists are in chronological order, which matters
for the chronological cross-validation folds.
"""
import glob
import os

HF_REPO = "Ouroboros/LOVW-8.5K"
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "LOVW-8.5K")

# The 17 sessions used in the paper. ME = motor execution (writing), MI = motor imagery.
SESSIONS = {
    "P1": {"ME": ["P1_S1_ME", "P1_S2_ME", "P1_S3_ME", "P1_S5_ME"],
           "MI": ["P1_S4_MI_1", "P1_S4_MI_2", "P1_S6_MI"]},
    "P2": {"ME": ["P2_S1_ME", "P2_S2_ME", "P2_S3_ME", "P2_S5_ME"],
           "MI": ["P2_S4_MI", "P2_S4_MI_index"]},
    "P3": {"ME": ["P3_S1_ME"], "MI": ["P3_S2_MI"]},
    "P4": {"ME": ["P4_S1_ME"], "MI": ["P4_S2_MI"]},
}
PARTICIPANTS = list(SESSIONS)

# ICA-cleaned, 0.3-40 Hz, 100 Hz EEGLAB files (the data used in the paper).
FILE_PATTERN = "{session}/preprocessed/0.3_40/*comp*elected*.set"


def download_patterns(sessions=None):
    """allow_patterns for huggingface_hub.snapshot_download (the .set and .fdt of each session)."""
    if sessions is None:
        sessions = [s for p in SESSIONS.values() for lst in p.values() for s in lst]
    return [f"{s}/preprocessed/0.3_40/*comp*elected*" for s in sessions]


def session_file(session, data_root=DEFAULT_DATA_ROOT):
    """Path of the .set file of a session (raises if missing or ambiguous)."""
    matches = sorted(glob.glob(os.path.join(data_root, FILE_PATTERN.format(session=session))))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file matching {FILE_PATTERN.format(session=session)} under {data_root}, "
            f"found {len(matches)}. Run `python scripts/download_data.py` first.")
    return matches[0]
