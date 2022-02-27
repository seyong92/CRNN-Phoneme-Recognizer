import torch
from torch.nn import functional as F
from torch import nn
from .feature import FeatureExtractor


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, data):
        x = self.cnn(data)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size).to(x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class CRNN(nn.Module):
    def __init__(self, config, consts):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.input_features = config.n_mels if config.feature_type == 'melspec' else config.n_mfcc
        self.output_features = config.num_lbl

        self.model = self._create_model(self.input_features,
                                        self.output_features, config)

        # self.melspec = MelSpectrogram(sr=consts.sample_rate, n_mels=config.n_mels)
        self.feat_ext = FeatureExtractor(config)

    def _create_model(self, input_features, output_features, config):
        modules = []
        model_complexity = config.model_complexity
        model_size = model_complexity * 16

        modules.append(ConvStack(input_features, model_size))
        modules.append(BiLSTM(model_size, model_size // 2))
        modules.append(nn.Linear(model_size, output_features))

        return nn.Sequential(*modules)

    def forward(self, data):
        return self.model(data)

    def run_on_batch(self, batch, cal_loss=True):
        feat = self.feat_ext(batch['audio']).transpose(1, 2).unsqueeze(1)  # (N, 1, T, F)
        pred = self(feat)  # (N, T, F)

        predictions = {
            'frame': pred,
        }

        if cal_loss:
            pred = pred.view(-1, self.output_features)  # (N * T, F)
            lbl = batch['label'].view(-1)

            # if self.sil_label:
            #     _, frame_lbl = batch['frame'].view(-1, self.output_features).max(dim=1)
            # else:
            #     frame_lbl = batch['frame'][:, :, 1:].view(-1, self.output_features)

            losses = {
                'loss': self.criterion(pred, lbl),
            }

            return predictions, losses

        return predictions
