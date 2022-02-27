import torch
import numpy as np
from collections import defaultdict

from mir_eval import melody
from mir_eval import transcription
from mir_eval.util import midi_to_hz
from tqdm import tqdm

from torch.utils.data import DataLoader
# from .decoding import extract_notes, notes_to_freqs
from .load_save import create_dict_raw, create_dict_compressed

import matplotlib
from matplotlib import pyplot as plt


def evaluate(dataset, model, config, consts, paths):
    metrics = defaultdict(list)

    for i, data in enumerate(tqdm(dataset)):
        data['audio'] = data['audio'].unsqueeze(0)
        data['label'] = data['label'].unsqueeze(0)
        preds, losses = model.run_on_batch(data)
        for k, v in losses.items():
            metrics[k].append(v)

        raw_dict_rev = create_dict_raw(paths[f'phoneset{config.num_lbl}_path'], reverse=True)
        dict_5 = create_dict_compressed(paths.phoneset5_path)
        dict_3 = create_dict_compressed(paths.phoneset3_path)

        _, phoneme_pred = torch.max(preds['frame'].squeeze().softmax(1), dim=1)
        phoneme_pred = phoneme_pred.squeeze()

        phoneme_pred_5 = torch.zeros_like(phoneme_pred)
        lbl_5 = torch.zeros_like(data['label'].squeeze())
        for j in range(phoneme_pred_5.size(0)):
            phoneme_pred_5[j] = dict_5[raw_dict_rev[int(phoneme_pred[j])]]
            lbl_5[j] = dict_5[raw_dict_rev[int(data['label'][0, j])]]

        phoneme_pred_3 = torch.zeros_like(phoneme_pred)
        lbl_3 = torch.zeros_like(data['label'].squeeze())
        for j in range(phoneme_pred_3.size(0)):
            phoneme_pred_3[j] = dict_3[raw_dict_rev[int(phoneme_pred[j])]]
            lbl_3[j] = dict_3[raw_dict_rev[int(data['label'][0, j])]]

        f1_macro, f1_micro, accuracy = _framewise_metrics(phoneme_pred,
                                                          data['label'].squeeze(), config.num_lbl)
        f1_macro_5, f1_micro_5, accuracy_5 = _framewise_metrics(phoneme_pred_5, lbl_5, 5)
        f1_macro_3, f1_micro_3, accuracy_3 = _framewise_metrics(phoneme_pred_3, lbl_3, 3)

        metrics['f1_macro'].append(f1_macro)
        metrics['f1_micro'].append(f1_micro)
        metrics['accuracy'].append(accuracy)
        metrics['f1_macro_5'].append(f1_macro_5)
        metrics['f1_micro_5'].append(f1_micro_5)
        metrics['accuracy_5'].append(accuracy_5)
        metrics['f1_macro_3'].append(f1_macro_3)
        metrics['f1_micro_3'].append(f1_micro_3)
        metrics['accuracy_3'].append(accuracy_3)

    return metrics


def _framewise_metrics(pred, lbl, num_lbl):
    eps = torch.finfo(torch.float).eps
    tp = torch.zeros(num_lbl)
    tn = torch.zeros(num_lbl)
    fp = torch.zeros(num_lbl)
    fn = torch.zeros(num_lbl)

    for i in range(pred.size(0)):
        if pred[i] == lbl[i]:
            tp[lbl[i]] += 1
            tn += 1
            tn[lbl[i]] -= 1
        else:
            fp[pred[i]] += 1
            fn[lbl[i]] += 1
            tn += 1
            tn[pred[i]] -= 1
            tn[lbl[i]] -= 1

    precision_macro = torch.mean(tp / (tp + fp + eps))
    recall_macro = torch.mean(tp / (tp + fn + eps))
    precision_micro = torch.sum(tp) / (torch.sum(tp) + torch.sum(fp) + eps)
    recall_micro = torch.sum(tp) / (torch.sum(tp) + torch.sum(fn) + eps)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro + eps)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)
    accuracy = torch.sum(tp) / pred.size(0)

    return f1_macro, f1_micro, accuracy


def _evaluate_song(p_ref, i_ref, t_ref, f_ref, p_est, i_est, t_est, f_est,
                   onset_tolerance, offset_min_tolerance):

    notes_metrics = transcription.precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est,
                                                              onset_tolerance=onset_tolerance,
                                                              offset_min_tolerance=offset_min_tolerance,
                                                              offset_ratio=None)
    notes_with_offsets_metrics = transcription.precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est,
                                                                           onset_tolerance=onset_tolerance,
                                                                           offset_min_tolerance=offset_min_tolerance)
    onset_metrics = transcription.onset_precision_recall_f1(i_ref, i_est,
                                                            onset_tolerance=onset_tolerance)
    offset_metrics = transcription.offset_precision_recall_f1(i_ref, i_est,
                                                              offset_min_tolerance=offset_min_tolerance)
    frames_metrics = melody.evaluate(t_ref, f_ref, t_est, f_est)

    return notes_metrics, notes_with_offsets_metrics, onset_metrics, offset_metrics, frames_metrics


def _decode_preds_and_lbls(data, model, config, consts):
    feat = data['feature'].to(config.device)
    lbl = dict()

    lbl['onset'] = data['onset']
    lbl['offset'] = data['offset']
    lbl['frame'] = data['frame']

    p_est, i_est, losses = model.transcribe(feat, lbl)

    lbl['onset'] = lbl['onset'][:, 1:]
    lbl['offset'] = lbl['offset'][:, 1:]
    lbl['frame'] = lbl['frame'][:, 1:]

    p_ref, i_ref = extract_notes(lbl['onset'], lbl['frame'], min_midi=config.min_midi)

    scale_factor = consts.hop_size / consts.sample_rate

    t_ref, f_ref = notes_to_freqs(p_ref, i_ref, lbl['frame'].size(0), scale_factor)
    t_est, f_est = notes_to_freqs(p_est, i_est, lbl['frame'].size(0), scale_factor)

    i_ref = (i_ref * scale_factor).reshape(-1, 2)
    p_ref = np.array([midi for midi in p_ref])
    i_est = (np.array(i_est) * scale_factor).reshape(-1, 2)
    p_est = np.array([midi for midi in p_est])

    t_ref = t_ref.astype(np.float64)
    t_est = t_est.astype(np.float64)

    return p_ref, i_ref, t_ref, f_ref, p_est, i_est, t_est, f_est, losses


def visualize_with_feature(batch, model, config, consts, paths):
    batch['audio'] = batch['audio'].unsqueeze(0)
    batch['label'] = batch['label'].unsqueeze(0)
    feat = model.feat_ext(batch['audio']).squeeze()  # (F, T)

    raw_dict_rev = create_dict_raw(paths.phoneset61_path, reverse=True)
    dict_5 = create_dict_compressed(paths.phoneset5_path)
    dict_3 = create_dict_compressed(paths.phoneset3_path)

    pred = model.run_on_batch(batch, cal_loss=False)['frame'].squeeze()  # (T, F)
    _, phoneme_pred = torch.max(pred.softmax(1), dim=1)

    phoneme_pred_5 = torch.zeros_like(phoneme_pred)
    lbl_5 = torch.zeros_like(batch['label'].squeeze())
    for i in range(phoneme_pred_5.size(0)):
        phoneme_pred_5[i] = dict_5[raw_dict_rev[int(phoneme_pred[i])]]
        lbl_5[i] = dict_5[raw_dict_rev[int(batch['label'][0, i])]]

    idx = pred.argmax(dim=1)
    frame_pred = torch.zeros_like(pred).scatter_(1, idx.unsqueeze(1), 1.)
    frame_lbl = torch.zeros_like(pred).scatter_(1, batch['label'].squeeze().unsqueeze(1), 1.)

    frame_pred_5 = torch.zeros(pred.size(0), 5).to(frame_pred.device).scatter_(1, phoneme_pred_5.unsqueeze(1), 1.)
    frame_lbl_5 = torch.zeros(pred.size(0), 5).to(frame_pred.device).scatter_(1, lbl_5.squeeze().unsqueeze(1), 1.)

    # phoneme_gt = [rev_dict[int(x)] for x in batch['phoneme']]

    # sr = consts.sample_rate

    plt.rcParams['figure.figsize'] = (4 * batch['audio'].size(1) / config.sample_rate, 20)

    frame_cmap = matplotlib.cm.get_cmap("Greys_r").copy()
    frame_cmap.set_under(color="white", alpha="0")
    frame_cmap.set_over(color='white', alpha="1")

    fig, (ax_gt, ax_pred, ax_gt_5, ax_pred_5) = plt.subplots(4, 1)

    ax_gt_lbl = ax_gt.twinx()
    ax_gt.imshow(feat.cpu().numpy(), aspect='auto', origin='lower')
    ax_gt_lbl.imshow(frame_lbl.cpu().numpy().T, aspect='auto', interpolation='none',
                       origin='lower', alpha=0.5, vmin=0.49, vmax=0.5, cmap=frame_cmap)

    # for i, note in enumerate(batch['notes']):
    #     ax_gt_lbl.text(note[0] * sr // config.hop_length, note[2] - consts.min_midi - 0.3,
    #                    batch['multiphone'][i], fontsize=15)

    ax_gt_lbl.hlines(np.arange(0.5, config.num_lbl + 0.5),
                     0, feat.size(1) - 1, colors='black', linestyles='dashed')

    ax_pred_lbl = ax_pred.twinx()
    ax_pred.imshow(frame_pred.cpu().numpy().T, aspect='auto', origin='lower')
    ax_pred_lbl.imshow(frame_pred.cpu().numpy().T, aspect='auto', interpolation='none',
                       origin='lower', alpha=0.5, vmin=0.49, vmax=0.5, cmap=frame_cmap)

    ax_pred_lbl.hlines(np.arange(0.5, config.num_lbl + 0.5),
                       0, feat.size(1) - 1, colors='black', linestyles='dashed')

    ax_gt_5.imshow(frame_lbl_5.cpu().numpy().T, aspect='auto', interpolation='none', origin='lower')
    ax_pred_5.imshow(frame_pred_5.cpu().numpy().T, aspect='auto', interpolation='none', origin='lower')

    # plt.yticks(np.arange(frame_pred.size(1)), [rev_dict[int(x)] for x in np.arange(frame_pred.size(1))])

    return fig
