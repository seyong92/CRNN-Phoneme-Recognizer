import torch
from phonerec.utils import load_model
import soundfile as sf
import librosa
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('audio_path', type=str)
    parser.add_argument('save_path', type=str)
    args = parser.parse_args()

    model, config, consts = load_model(args.model_path)
    model.eval()

    x, sr = sf.read(args.audio_path)
    if x.ndim > 1:
        x = x[:, 0]
    x = librosa.resample(x, sr, config.sample_rate)
    x_tensor = torch.from_numpy(x).float().to(config.device)

    with torch.no_grad():
        batch = dict()
        batch['audio'] = x_tensor.unsqueeze(0)
        predictions = model.run_on_batch(batch, cal_loss=False)

    torch.save(predictions['frame'].detach().to('cpu'), args.save_path)
