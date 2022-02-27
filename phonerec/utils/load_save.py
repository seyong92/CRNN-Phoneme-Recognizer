from phonerec.models import *
import yaml
from attributedict.collections import AttributeDict
import torch
from torch.utils.data import ConcatDataset
from ..datasets import PhonemeFrameDataset
import wandb
from sys import argv
import re


def load_yaml(filename):
    with open(filename) as f:
        yml_dict = yaml.load(f, Loader=yaml.SafeLoader)
        yml_dict = AttributeDict(yml_dict)

    return yml_dict


def load_datasets(dataset_names, config, paths, group=None, debug=False):
    dataset_list = []
    for dataset_name in dataset_names:
        path = paths[f'{dataset_name.lower()}_path_prep']
        dataset = PhonemeFrameDataset(config, path, group, debug)
        dataset_list.append(dataset)

    if len(dataset_list) == 1:
        return dataset_list[0]
    else:
        return ConcatDataset(dataset_list)


def create_model(config, consts, device=None):
    model = globals()[config.model](config, consts)

    if device is not None:
        model.to(device)
    else:
        model.to(config.device)

    return model


def save_model(save_path, model, config, consts):
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_state_dict = model.state_dict()
        save_data = {'model_state_dict': model_state_dict,
                     'config': config.to_dict(),
                     'consts': consts.to_dict()}

        torch.save(save_data, save_path)


def load_model(file):
    save_data = torch.load(file, map_location='cpu')
    model_state_dict = save_data['model_state_dict']
    config = AttributeDict(save_data['config'])
    consts = AttributeDict(save_data['consts'])

    model = create_model(config, consts, 'cpu')
    model.load_state_dict(model_state_dict)
    model.to(config.device)

    return model, config, consts


def parse_argv():
    arg_dict = dict()
    argc = len(argv)
    argv_list = []
    for i in range(argc):
        argv_list.append(argv[i].split('='))

    argv_list = [item for sublist in argv_list for item in sublist]

    for i in range(1, len(argv_list), 2):
        arg_dict[argv_list[i].lstrip('-').replace('-', '_')] = _arg_type_fix(argv_list[i + 1])

    return arg_dict


def _arg_type_fix(arg):
    try:
        int(arg)
        return int(arg)
    except ValueError:
        try:
            float(arg)
            return float(arg)
        except ValueError:
            return arg


def create_dict_raw(phoneset_path, reverse=False):
    dict_raw = dict()
    line_idx = 0
    with open(phoneset_path, 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            if reverse:
                dict_raw[line_idx] = phoneme
            else:
                dict_raw[phoneme] = line_idx
            line_idx += 1

    return dict_raw


def create_dict_compressed(phoneset_path, reverse=False):
    dict_comp = dict()
    line_idx = 0
    with open(phoneset_path, 'r') as f:
        for line in f:
            phonemes = re.split('\s+|,', line.strip())
            if reverse:
                dict_comp[line_idx] = phonemes[0]
            else:
                for phoneme in phonemes:
                    dict_comp[phoneme] = line_idx
            line_idx += 1

    return dict_comp


def save_log(log_dict, global_step, writer=None, print_log=True):
    if writer is None:
        wandb.log(log_dict, step=global_step)
    else:
        for key, value in log_dict.items():
            writer.add_scalar(key, value, global_step=global_step)

    if print_log:
        print(log_dict)
