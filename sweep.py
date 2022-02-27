from phonerec.utils import parse_argv, load_yaml
from train import train


if __name__ == '__main__':
    config = load_yaml('configs/config.yaml')
    config_arg = parse_argv()
    config_model = load_yaml(config_arg['model_config'])

    config.update(config_model)
    config.update(config_arg)

    consts = load_yaml('configs/constants.yaml')
    paths = load_yaml('configs/paths.yaml')

    train('wandb', config, consts, paths)
