import os
import yaml
import re


def load_configs(configs_path="./configs/configs.yml"):
    with open(configs_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    return configs
