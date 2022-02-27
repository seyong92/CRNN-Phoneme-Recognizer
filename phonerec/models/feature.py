from numpy import fmin
import torch
from torch.nn import functional as F
from torch import nn
from torchaudio import transforms as T
from nnAudio import Spectrogram as S


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.feature_type == 'spec':
            self.feat = S.STFT(sr=config.sample_rate,
                               n_fft=config.win_length,
                               win_length=config.win_length,
                               hop_length=config.hop_length,
                               output_format='Magnitude').to(config.device)
            self.db = T.AmplitudeToDB(stype='magnitude', top_db=80)
        elif config.feature_type == 'melspec':
            self.feat = S.MelSpectrogram(sr=config.sample_rate,
                                         n_fft=config.win_length,
                                         win_length=config.win_length,
                                         n_mels=config.n_mels,
                                         hop_length=config.hop_length,
                                         fmin=config.fmin,
                                         fmax=config.fmax,
                                         center=False).to(config.device)
            self.db = T.AmplitudeToDB(stype='power', top_db=80)
        elif config.feature_type == 'mfcc':
            self.feat = S.MFCC(sr=config.sample_rate,
                               n_mfcc=config.n_mfcc,
                               n_fft=config.win_length,
                               win_length=config.win_length,
                               n_mels=config.n_mels,
                               hop_length=config.hop_length,
                               fmin=config.fmin,
                               fmax=config.fmax).to(config.device)
            self.db = None

    def forward(self, audio):
        feature = self.feat(audio)
        if self.db is not None:
            feature = self.db(feature)

        return feature
