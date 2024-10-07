
"""
# Data Pipline Utilities

Contains useful functions for data creating, loading and saving stuff.

"""


from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import pickle
import torch
import json

from codex import ok, error, warning, log



class Dataset:
    
    """Main class for creating, loading, saving and batchifying data."""
    
    index = 0
    def __init__(self, name : str = None, **metadata) -> None:
        self.name = name
        if name is None:
            self.name = f'dataset_{Dataset.index}'
            Dataset.index += 1
        
        self.metadata = {
            'name' : self.name,
            **metadata
        }
    
    def create(
        self,
        path : str = '.',
        image_size : tuple[int, int] = (128, 128),
        channels : int = 3
    ) -> None:
        """Creates a dataset, reading from files. Image files are assumed to be @ `./image_data` 
        and organised under `dataset_real_{dindex}` directories. CSV files are assumed to be @ `./csv_files` 
        and named `dataset_real_{dindex}.csv`. The images are normalised to 0.0 to 1.0.
        
        Params:
            path : Path to the root of the dataset directory.
            dindex : The dataset to build from.
            image_size : Size of images to downsample to.
        """
        rootpath = path
        if os.path.exists(os.path.join(rootpath, 'config.json')):
            with open(os.path.join(rootpath, 'config.json'), 'r') as file:
                config = json.load(file)
                print('Config file:', end='')
                log(config)
            for key in ['rate', 'type', 'linear_vel', 'angular_vel']:
                self.metadata[key] = config[key]
        else:
            warning(f'No config file found for @ {rootpath}...')
            warning('Assuming type is `color/image_raw`')
            self.metadata['type'] = 'color/image_raw'
        
        data_type = self.metadata['type']
        
        data_path = os.path.join(rootpath, *data_type.split('/'))
        
        n = len(os.listdir(data_path))
        self.metadata['samples'] = n
        fulldatasetx = np.zeros((n, *image_size, channels), dtype=np.float32)
        
        for each in range(n):
            print(f'\rReading image {each + 1}/{n}...', end='')
            ipath = os.path.join(data_path, f'{each}.png')
            fulldatasetx[each] = cv.resize(plt.imread(ipath), dsize=image_size, interpolation=cv.INTER_LINEAR)
        
        joint_states = pd.read_csv(os.path.join(rootpath, 'joints.csv'))
        joint_positions = joint_states.loc[:, [f'joint_{i}_position' for i in range(7)]].to_numpy()
        joint_velocities = joint_states.loc[:, [f'joint_{i}_velocity' for i in range(7)]].to_numpy()
        
        gripper_states = pd.read_csv(os.path.join(rootpath, 'gripper.csv')).to_numpy()
        
        self.x_data = fulldatasetx
        self.y_pos = joint_positions #[: 2 * n : 2]
        self.y_vel = joint_velocities #[: 2 * n : 2]
        print()
        ok(f'Created dataset {self.name}!')
        
        self.metadata['size'] = image_size
    
    def save(self, name_or_path : str) -> None:
        """Saves a dataset to disk."""
        with open(f'{name_or_path}.rdata', 'wb') as file:
            pickle.dump((self.metadata, (self.x_data, self.y_pos, self.y_vel)), file)
        print(f'Saved dataset {self.name} @ {name_or_path}.rdata')

    def load(self, path : str, quiet : bool = False) -> None:
        """Loads an existing pickled dataset."""
        with open(path, 'rb') as file:
            meta, (xs, ypos, yvel) = pickle.load(file)
        self.metadata = meta
        self.x_data = xs
        self.y_pos = ypos
        self.y_vel = yvel
        if 'name' in meta: self.name = meta['name']
        if 'samples' in meta: self.samples = meta['samples']
        else: self.samples = len(self.x_data)
        
        if not quiet:
            print(f'Loaded dataset {self.name}: ', end='')
            log(self.metadata)
        
    
    def _split(self, train_test_validation : tuple[int, int, int] = (7, 2, 1)) -> None:
        """
        DOESNT WORK YET.
        Splits the dataset into train, test and validation splits based on the given ratio."""
        #TODO: Think about this and implement it.
        
        trainratio = train_test_validation[0] / sum(train_test_validation)
        testratio = train_test_validation[1] / sum(train_test_validation)
        
        trainlen = int(trainratio * len(self.x_data))
        testlen = int(testratio * len(self.x_data))
        
        self.trainx = self.x_data[:trainlen]
        self.testx = self.x_data[trainlen : trainlen + testlen]
        self.validationx = self.x_data[trainlen + testlen:]
        
        self.trainy = self.y_vel[:trainlen]
        self.testy = self.y_vel[trainlen : trainlen + testlen]
        self.validationy = self.y_vel[trainlen + testlen:]
        
        
    
    def get_batch(
            self,
            batch_size : int,
            time_steps : int = 32,
            include_only_last_y : bool = True,
            split : Literal['train', 'test', 'valid'] = 'train'
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes and returns a single batch, can be used when training. The batch will contain contiguous sequences from 
        the x's and y's in the dataset. Make sure to call `.split()` to get train-test-validation splits before this.

        Args:
            batch_size (int): size of the batch
            time_steps (int, optional): No. of contiguous x's (and y's depending on `include_only_last_y`) per datapoint in the batch. Defaults to 32.
            include_only_last_y (int, optional): If True, only last y value is included per datapoint. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: batch_x, batch_y
        
        The shape of `xs` is (batch_dim, channels, time_window, height, width).
        
        The shape of `ys` is (batch_dim, no. of joints (7)).
        """
        batchx, batchy = [], []
        xs, ys = self.trainx, self.trainy
        if split == 'test':
            xs, ys = self.testx, self.testy
        if split == 'valid':
            xs, ys = self.validationx, self.validationy
        
        indices = np.random.randint(0, len(xs) - time_steps, batch_size)
        for each in range(batch_size):
            batchx.append(xs[indices[each] : indices[each] + time_steps])
            if include_only_last_y:
                batchy.append(ys[indices[each] + time_steps - 1])
            else:
                batchy.append(ys[indices[each] : indices[each] + time_steps])
        
        batchx = np.array(batchx)
        batchy = np.array(batchy)
        
        batchx = np.transpose(batchx, (0, 4, 1, 2, 3))
        
        batchx = torch.tensor(batchx, dtype=torch.float32)
        batchy = torch.tensor(batchy, dtype=torch.float32)
        
        return (batchx, batchy)


def load_dataset(path : str, quiet : bool = False) -> Dataset:
    """Loads a dataset from a `.rdata` dataset file."""
    dataset = Dataset()
    dataset.load(path, quiet = quiet)
    return dataset




if __name__ == '__main__':
    d = Dataset('bottle_data', description='Picking up and placing a bottle')
    d.create('dataset3')
    d.save('bottle_data')
    

