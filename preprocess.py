import torch
import soundfile as sf
from tqdm import tqdm
import numpy as np
from pathlib import Path

from phonerec.utils import load_yaml, create_dict_compressed, create_dict_raw


def process(phn_file, wav_file, dict61, dict39, dict5, dict3):
    x, _ = sf.read(str(wav_file))

    x_61, seq_61 = _phn2vec(x.size, phn_file, dict61)
    x_39, seq_39 = _phn2vec(x.size, phn_file, dict39)
    x_5, seq_5 = _phn2vec(x.size, phn_file, dict5)
    x_3, seq_3 = _phn2vec(x.size, phn_file, dict3)

    return dict(audio=torch.from_numpy(x).float().clone(),
                label_61=torch.from_numpy(x_61).type(torch.int8).clone(),
                seq_61=torch.Tensor(seq_61).type(torch.int8).clone(),
                label_39=torch.from_numpy(x_39).type(torch.int8).clone(),
                seq_39=torch.Tensor(seq_39).type(torch.int8).clone(),
                label_5=torch.from_numpy(x_5).type(torch.int8).clone(),
                seq_5=torch.Tensor(seq_5).type(torch.int8).clone(),
                label_3=torch.from_numpy(x_3).type(torch.int8).clone(),
                seq_3=torch.Tensor(seq_3).type(torch.int8).clone())


def _phn2vec(wav_length, phn_path, phn_dict):
    phone_time_seq = np.zeros(wav_length)
    phone_seq = []
    with open(str(phn_path)) as f:
        for line in f:
            sample_start, sample_end, label_char = line.split(' ')
            sample_start = int(sample_start)
            sample_end = int(sample_end)
            label_char = label_char.strip().replace('-', '')
            # if label_char == 'q':
            #     continue
            label_num = phn_dict[label_char]

            phone_time_seq[sample_start: sample_end] = label_num
            phone_seq.append(label_num)

    return phone_time_seq, phone_seq


def preprocess_timit(consts, paths):
    orig_path = Path(paths.timit_path)
    prep_path = Path(paths.timit_path_prep)

    dict61 = create_dict_raw(paths.phoneset61_path)
    dict39 = create_dict_compressed(paths.phoneset39_path)
    dict5 = create_dict_compressed(paths.phoneset5_path)
    dict3 = create_dict_compressed(paths.phoneset3_path)

    if not prep_path.exists():
        prep_path.mkdir()

    for group in ['train', 'test']:
        group_path = orig_path / group.upper()

        phns = list(group_path.glob('**/*.phn')) + list(group_path.glob('**/*.PHN'))
        wavs = list(group_path.glob('**/*.wav')) + list(group_path.glob('**/*.WAV'))
        phns.sort()
        wavs.sort()

        fnames = []

        for i in tqdm(range(len(phns))):
            prep = process(phns[i], wavs[i], dict61, dict39, dict5, dict3)

            file_name_base = f'{wavs[i].parent.stem}_{wavs[i].stem}'
            fnames.append(file_name_base)

            torch.save(prep, prep_path / (file_name_base + '.pt'))

        with open(prep_path / f'{group}_list.txt', 'w') as f:
            for fname in fnames:
                f.write(fname + '\n')


if __name__ == '__main__':
    consts = load_yaml('configs/constants.yaml')
    paths = load_yaml('configs/paths.yaml')
    config = load_yaml('configs/config.yaml')

    preprocess_timit(consts, paths)
