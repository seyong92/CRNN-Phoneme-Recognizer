CRNN-Phoneme-Recognizer
=======================
PyTorch implementation of frame-level CRNN phoneme recognizer trained with TIMIT.

## Installation
Install Python version higher than 3.7. and PyTorch version higher than 1.10.1.

If you use [poetry](https://python-poetry.org), you can install other packages easily through the following command.

```
$ poetry install
```

If you don't use poetry, please just install packages in ```pyproject.toml``` manually.

## Extract phoneme posteriorgram (PPG) with pretrained model.
You can download pretrained model from [Releases](https://github.com/seyong92/CRNN-Phoneme-Recognizer/releases/tag/v0.1.0), and you can save PPG through the following command.
```
$ python infer.py [MODEL_PATH] [AUDIO_PATH] [SAVE_PATH]
```

## Train your own model
If you want to train your own model, you should prepare TIMIT dataset, and fix the ```configs/paths.yaml``` to your TIMIT dataset path.

Then, you can train the model through the following command.
```
$ python train.py [LOGGER]
```
LOGGER can be wandb, tensorboard, none (do not log anything for debug). If you want to fix the hyperparameter, you can fix ```configs/models/CRNN.yaml```.