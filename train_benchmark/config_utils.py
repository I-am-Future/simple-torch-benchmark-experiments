'''
Config utilities for training.
Author: Lai Wei
Mar. 31st, 2023
'''

import os
import logging
import time
from torch.utils.tensorboard import SummaryWriter

class CONFIG:
    ''' A general class, which can record configs, do logging to the file, and tensorboard'''
    def __init__(self, configs: dict):
        # save the configs
        self._config = configs
        
        # maybe only good for linux/macos
        logfile = os.path.join(self._config['work_dir'], f'{time.ctime()}.log')

        file_handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        self._logger = logger

        if configs.get('enable_tb', False):
            self.log_string('We have tensorboard enabled.')
            self._writer = SummaryWriter(self._config['work_dir'])

        self.log_string('Initializing configs...')
        self.log_string(f'The configs are: {configs}')

    def log_string(self, content, do_print: bool=True):
        self._logger.info(content)
        if do_print:
            print(content)

    def log_tensorboard_scaler(self, tag: str, value: float, step: int):
        if self._config.get('enable_tb', False):
            self._writer.add_scalar(tag, value, step)
        else:
            # we don't enable tensorboard here
            raise UserWarning('We have NOT enabled TensorBoard Currently. Check the config for "enable_tb".')

    def __getitem__(self, key):
        return self._config[key]

def count_params(model, only_trainable: bool=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        sum(p.numel() for p in model.parameters())