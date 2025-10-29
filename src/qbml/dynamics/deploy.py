import os
from pathlib import Path


def deploy_it(name, loc):
    if loc == '.':
        dir_path = Path(os.getcwd()) / Path(name)
    else:
        dir_path = Path(loc + 'qmlproject')
    config_path = dir_path / 'configs'
    setgen_cfg_path = config_path / 'setgen'
    train_cfg_path = config_path / 'training'
    data_path = dir_path / 'data'
    model_path = dir_path / 'models'
    committee_path = dir_path / 'committees'
    logs_path = dir_path / 'logs'
    paths = [dir_path,
             config_path,
             setgen_cfg_path,
             train_cfg_path,
             data_path,
             model_path,
             committee_path,
             logs_path]
    for path in paths:
        os.mkdir(path)
    
