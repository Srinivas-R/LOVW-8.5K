"""Decoders used in the paper."""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018), as used in the paper.

    Input: (batch, channels, time). F1 temporal filters of length sampling_rate/2, D spatial filters each,
    separable convolution, two average-pooling stages (4 and 5) and a linear classifier.
    Note: BatchNorm eps is 0, exactly as in the experiments reported in the paper.
    """

    def __init__(self, n_channels=32, n_times=100, sampling_rate=100, n_outputs=4, F1=8, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.conv1 = nn.Conv2d(1, F1, (1, sampling_rate // 2), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, eps=0.0)
        self.conv2 = nn.Conv2d(F1, F2, (n_channels, 1), padding='valid', groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F2, eps=0.0)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(nn.Conv2d(F2, F2, (1, n_times // 8), padding='same', groups=F2, bias=False),
                                   nn.Conv2d(F2, F2, (1, 1), padding='same', bias=False))
        self.batchnorm3 = nn.BatchNorm2d(F2, eps=0.0)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 5))
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(F2 * n_times // 20, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)                                   # (B, 1, C, T)
        x = self.dropout1(self.pooling1(self.activation1(self.batchnorm2(self.conv2(self.batchnorm1(self.conv1(x)))))))
        x = self.dropout2(self.pooling2(self.activation2(self.batchnorm3(self.conv3(x)))))
        return self.fc1(x.flatten(1))


def build_braindecode(name, n_channels, n_times, sfreq, n_outputs=4):
    """Specialist baselines from braindecode (reimplemented for this release; see README)."""
    from braindecode.models import EEGConformer, SPARCNet
    if name == 'conformer':
        return EEGConformer(n_chans=n_channels, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq,
                            final_fc_length='auto')
    if name == 'sparcnet':
        return SPARCNet(n_chans=n_channels, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    raise ValueError(name)


class CBraModClassifier(nn.Module):
    """CBraMod backbone (Wang et al., 2025) with a linear head on the flattened patch features.

    Input: (batch, channels, time) at 200 Hz in units of 100 uV; time must be a multiple of 200 (1 s patches).
    """

    def __init__(self, weights_path, n_channels=32, n_times=800, n_outputs=4, linear_probe=False):
        super().__init__()
        from .third_party.cbramod import CBraMod
        self.backbone = CBraMod()
        self.backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.backbone.proj_out = nn.Identity()
        self.to_patches = Rearrange('b c (s p) -> b c s p', p=200)
        self.head = nn.Linear(n_channels * (n_times // 200) * 200, n_outputs)
        if linear_probe:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.head(self.backbone(self.to_patches(x)).flatten(1))


def cbramod_weights(cache_dir=None):
    """Download the official CBraMod pretrained weights (https://huggingface.co/weighting666/CBraMod)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download('weighting666/CBraMod', 'pretrained_weights.pth', cache_dir=cache_dir)


class REVEClassifier(nn.Module):
    """REVE backbone (brain-bzh/reve-base) with a linear head on the flattened patch features.

    Loads the official model from Hugging Face, which requires `trust_remote_code=True` (code from the REVE
    authors). Input: (batch, channels, time) at 200 Hz in units of 100 uV, time a multiple of 200.
    """

    def __init__(self, electrodes, n_times=800, n_outputs=4, linear_probe=False):
        super().__init__()
        from transformers import AutoModel
        positions = AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained('brain-bzh/reve-base', trust_remote_code=True)
        with torch.no_grad():
            self.register_buffer('positions', positions(electrodes).float())   # (channels, 3)
        self.head = nn.Linear(len(electrodes) * (n_times // 200) * 512, n_outputs)
        if linear_probe:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        pos = self.positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.head(self.backbone(x, pos).flatten(1))
