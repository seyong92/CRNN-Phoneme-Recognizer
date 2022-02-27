import torch
from torch.utils.data import Dataset
from torch.nn.functional import conv1d, pad
from tqdm import tqdm
from pathlib import Path
from attributedict.collections import AttributeDict


class PhonemeFrameDataset(Dataset):
    def __init__(self, config, path, group=None, debug=False):
        self.path = Path(path)
        self.device = config.device
        self.num_frames = config.num_frames if group == 'train' else -1
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.num_lbl = config.num_lbl

        self.data = []

        data_fnames = tqdm(self._data_files(group, debug))

        for _, data_fname in enumerate(data_fnames):
            data_fnames.set_description(f'Loading {data_fname}')

            data = self._load_file(data_fname, config)
            self.data.append(data)

    def _data_files(self, group, debug=False):
        if group is None:
            file_list = [f.stem for f in self.path.glob('*.pt')]
        else:
            with open(self.path / f'{group}_list.txt') as f:
                file_list = f.read().splitlines()

        if debug:
            file_list = file_list[:100]

        return file_list

    def _load_file(self, fname, config):
        data = torch.load(self.path / f'{fname}.pt')

        win_eye = torch.eye(config.win_length).unsqueeze(1)

        label = data[f'label_{config.num_lbl}'].unsqueeze(0).unsqueeze(0)

        label = conv1d(label.float(), win_eye, stride=config.hop_length).squeeze()
        label, _ = torch.mode(label, dim=0)
        label = label.long()

        data.update({'label': label})

        return AttributeDict(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        result = dict()

        if self.num_frames != -1:
            data_frames = data['label'].size(0)
            num_samples = (data_frames - 1) * self.hop_length + self.win_length

            if data_frames >= self.num_frames:  # case for long audio
                if data_frames > self.num_frames:
                    begin = int(torch.randint(0, data_frames - self.num_frames, (1, )))
                else:
                    begin = 0
                end = begin + self.num_frames

                result['audio'] = data['audio'][begin * self.hop_length: (end - 1) * self.hop_length + self.win_length].to(self.device)
                result['label'] = data['label'][begin: end].to(self.device)
            else:  # case for short audio
                if self.num_lbl == 39:
                    sil = 29
                elif self.num_lbl == 61:
                    sil = 27
                elif self.num_lbl == 5:
                    sil = 4
                elif self.num_lbl == 3:
                    sil = 2

                l_pad = int(torch.randint(0, self.num_frames - data_frames, (1, )))
                r_pad = self.num_frames - data_frames - l_pad
                result['audio'] = pad(data['audio'].to(self.device),
                                      (l_pad * self.hop_length, r_pad * self.hop_length + num_samples - data['audio'].size(0)))
                result['label'] = pad(data['label'].to(self.device),
                                      (l_pad, r_pad), value=sil)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)

        return result
