import wandb
import torch
import argparse
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from pathlib import Path
from matplotlib import pyplot as plt
import os

from torch.utils import tensorboard

from phonerec.utils import load_yaml, load_datasets, create_model, load_model
from phonerec.utils import evaluate, visualize_with_feature
from phonerec.utils import save_log, save_model
from phonerec.utils import EarlyStop


def train(logger, config, consts, paths):
    # Initialize seeds
    if config.seed == -1:
        seed = torch.seed()
        config.update({'seed': seed})
    else:
        torch.manual_seed(config.seed)

    # Set logger
    if logger == 'wandb':
        run = wandb.init(config=config)
        writer = None
        log_dir = Path(wandb.run.dir)
        log_name = log_dir.relative_to(Path(os.getcwd()) / 'wandb').parent
        model_log_dir = Path('wandb_model_save') / log_name
        debug = False
    elif logger == 'tensorboard':
        writer = tensorboard.SummaryWriter()
        log_dir = Path(writer.log_dir)
        model_log_dir = log_dir
        debug = False
        print(f'tensorboard log dir: {log_dir}')
    elif logger == 'none':
        run = wandb.init(config=config, mode='disabled')
        writer = None
        log_dir = None
        model_log_dir = None
        debug = False
    config.update({'logger': logger})

    # Load dataset
    train_dataset = load_datasets(config.dataset.train, config, paths, 'train', debug)
    valid_dataset = load_datasets(config.dataset.valid, config, paths, 'valid', debug)
    test_dataset = load_datasets(config.dataset.test, config, paths, 'test', debug)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    # Load model and loss function, optimizer, etc.
    if config.resume_iteration == 0:
        model = create_model(config, consts)
        optim = torch.optim.Adam(model.parameters(), config.learning_rate)
    else:
        pass
    scheduler = StepLR(optim, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)
    early_stopper = EarlyStop(patience=config.patience, goal=config.goal,
                              delta=config.delta)
    stop = 'best'

    # Train model and evaulate
    loop = tqdm(range(config.resume_iteration, config.max_iteration))
    for i, batch in zip(loop, cycle(train_loader)):
        pred, losses = model.run_on_batch(batch)
        loss = sum(losses.values())
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        save_log({'loss/train': loss.item()}, i, writer, print_log=False)

        # Evaluation with valid dataset
        if i % config.valid_interval == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                valid_eval = evaluate(valid_dataset, model, config, consts, paths)
                for key, value in valid_eval.items():
                    mean = float(sum(value) / len(value))
                    save_log({'valid/' + key: mean}, i, writer)

                batch = dict()
                batch['audio'] = valid_dataset[0]['audio'].to(config.device)
                batch['label'] = valid_dataset[0]['label'].to(config.device)
                if logger == 'wandb':
                    fig = visualize_with_feature(batch, model, config, consts, paths)
                    wandb.log({'valid/example_0': fig}, step=i)
                plt.close()
            model.train()

            valid_loss = valid_eval[config.es_criteria]
            valid_loss = float(sum(valid_loss) / len(valid_loss))
            stop = early_stopper(valid_loss)

        # Save model
        if (i % config.valid_interval == 0) and (logger != 'none') and i > 0:
            save_model(model_log_dir / 'model-recent.pt', model, config, consts)
            torch.save(optim.state_dict(), model_log_dir / 'last-optimizer-state.pt')
            if stop == 'best':
                save_model(model_log_dir / 'model-best.pt', model, config, consts)

        # check early stopping
        if stop == 'stop':
            break

    # Evaluation with test dataset
    model, _, _ = load_model(model_log_dir / 'model-best.pt')
    model.eval()
    with torch.no_grad():
        for key, value in evaluate(test_dataset, model, config, consts, paths).items():
            mean = float(sum(value) / len(value))
            save_log({'test/' + key: mean}, i, writer)

        if logger == 'wandb':
            demo_idx = [10, 20, 30, 40, 50]
            for idx in demo_idx:
                batch = dict()
                batch['audio'] = test_dataset[idx]['audio'].to(config.device)
                batch['label'] = test_dataset[idx]['label'].to(config.device)

                fig = visualize_with_feature(batch, model, config, consts, paths)
                wandb.log({f'test/example_{idx}': fig}, step=i)
                plt.close()

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logger', choices=['wandb', 'tensorboard', 'none'])
    args = parser.parse_args()

    config = load_yaml('configs/config.yaml')
    config_model = load_yaml(config.model_config)
    config.update(config_model)
    consts = load_yaml('configs/constants.yaml')
    paths = load_yaml('configs/paths.yaml')

    print(config)

    train(args.logger, config, consts, paths)
