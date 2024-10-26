
"""
# Data Pipline Utilities

Contains useful functions for data creating, loading and saving stuff.

"""


from math import ceil
from typing import Generator, Literal
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import torch
import json

from codex import ok, error, warning, log
from .utils import talosdir

from safetensors import numpy as stnp
from safetensors import safe_open


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
        
        self.mask = {
            'train' : None,
            'test' : None,
            'valid' : None,
        }
        
        self.segments = {
            'train' : None,
            'test' : None,
            'valid' : None,
        }
        
        self.batch_index = None
        
        
        
    
    def create(
        self,
        path : str = '.',
        image_size : tuple[int, int] = (128, 128),
    ):
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
        self.samples = n
        # fulldatasetx = np.zeros((n, *image_size, channels), dtype=np.float32)
        
        fulldatasetx = []
        for each in range(n):
            print(f'\rReading image {each + 1}/{n}...', end='')
            ipath = os.path.join(data_path, f'{each}.png')
            fulldatasetx.append(
                cv.resize(plt.imread(ipath), dsize=image_size, interpolation=cv.INTER_LINEAR)
            )
        fulldatasetx = np.array(fulldatasetx)    
        
        joint_states = pd.read_csv(os.path.join(rootpath, 'joints.csv'))
        joint_states.columns = list(map(lambda x:x.strip(), joint_states.columns))
        joint_positions = joint_states.loc[:, [f'joint_{i}_position' for i in range(7)]].to_numpy()
        joint_velocities = joint_states.loc[:, [f'joint_{i}_velocity' for i in range(7)]].to_numpy()
        
        gripper_states = pd.read_csv(os.path.join(rootpath, 'gripper.csv')).to_numpy()
        
        self.x_data = fulldatasetx
        self.y_pos = joint_positions #[: 2 * n : 2]
        self.y_vel = joint_velocities #[: 2 * n : 2]
        print()
        ok(f'Created dataset {self.name}!')
        
        self.metadata['size'] = image_size
        
        return self
    
    
    
    def save(self, name_or_path : str = None) -> None:
        """Saves a dataset to disk."""
        if name_or_path is None:
            name_or_path = os.path.join(talosdir, 'datasets', self.name)
        dirr = os.path.dirname(name_or_path)
        if dirr: os.makedirs(dirr, exist_ok=True)
        
        meta = dict()
        for key in self.metadata:
            meta[key] = str(self.metadata[key])
        
        stnp.save_file({
            'x_data' : self.x_data,
            'y_pos' : self.y_pos,
            'y_vel' : self.y_vel,
        }, f'{name_or_path}.rdata', meta)
        
        
        print(f'Saved dataset {self.name} @ {name_or_path}.rdata')


    def load(self, name : str = None, path : str = None, quiet : bool = False) -> None:
        """Loads an existing saved dataset."""
        
        if path is not None: final = path
        elif name is not None: final = f'{os.path.join(talosdir, 'datasets', name)}.rdata'
            
        with safe_open(final, framework='numpy') as file:
            meta = file.metadata()
        data = stnp.load_file(final)
        xs = data['x_data']
        yvel = data['y_vel']
        ypos = data['y_pos']
            
                
        self.metadata = meta
        self.x_data = xs
        self.y_pos = ypos
        self.y_vel = yvel
        if 'name' in meta: self.name = meta['name']
        if 'samples' in meta: self.samples = int(meta['samples'])
        else: self.samples = len(self.x_data)
        
        if not quiet:
            print(f'Loaded dataset {self.name}: ', end='')
            log(self.metadata)
        
        return self
        
    
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
        
    
    def nbatches(
            self, 
            batch_size : int, 
            time_steps : int,
            split : Literal['train', 'test', 'valid'] = 'train',
        ) -> int:
        xs = self.trainx
        if split == 'test':
            xs = self.testx
        if split == 'valid':
            xs = self.validationx
            
        return ceil((len(xs) - time_steps) / batch_size)
    
    
    def get_batch_new_new(
            self,
            batch_size : int,
            time_steps : int = 32,
            split : Literal['train', 'test', 'valid'] = 'train',
        ) -> Generator:
        """Best possible batching I could come up with. Returns a generator using yield, 
        and is supposed to be used like so:
        
        ```
        for xb, yb in dataset.get_batch_new_new(16, 16):
            xb = xb.to('cuda')
            yb = yb.to('cuda')
        
            # Eg.
            ycap = model(xb)

        ```
        

        Args:
            batch_size (int): no. of samples in the batch
            time_steps (int, optional): size of context window. Defaults to 32.
            split (Literal[&#39;train&#39;, &#39;test&#39;, &#39;valid&#39;], optional): which split to take samples from. Defaults to 'train'.

        Yields:
            xb, yb (tuple[torch.Tensor, torch.Tensor])
        """
        xs, ys = self.trainx, self.trainy
        if split == 'test':
            xs, ys = self.testx, self.testy
        if split == 'valid':
            xs, ys = self.validationx, self.validationy
        
        ix = 0
        while (ix + time_steps) < len(xs):
            batchx = []
            batchy = []
            for _ in range(min(batch_size, len(xs) - ix - time_steps)):                
                batchx.append(xs[ix : ix + time_steps])
                batchy.append(ys[ix + time_steps])
                ix += 1
            
            batchx = torch.tensor(np.array(batchx, dtype=np.float32))
            batchy = torch.tensor(np.array(batchy, dtype=np.float32))
            
            yield batchx, batchy
        
        return None
        
        
        
        
    
    def get_batch_new(
            self,
            batch_size : int,
            time_steps : int = 32,
            split : Literal['train', 'test', 'valid'] = 'train',
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Better batching than `.get_batch()`.

        Args:
            batch_size (int): batch size.
            time_steps (int, optional): size of time window. Defaults to 32.
            include_only_last_y (bool, optional): if true, only last y value for each sample is included in ybatch. Defaults to True.
            split (Literal[&#39;train&#39;, &#39;test&#39;, &#39;valid&#39;], optional): which split to get batch from. Defaults to 'train'.
            non_zero_thresh (float, optional): threshold over which a y is counted as non-zero. Defaults to 1e-2.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (x-batch, y-batch)
        """
        xs, ys = self.trainx, self.trainy
        if split == 'test':
            xs, ys = self.testx, self.testy
        if split == 'valid':
            xs, ys = self.validationx, self.validationy
        
        
        if self.batch_index is None: self.batch_index = 0
        
        index = self.batch_index
        batchx, batchy = [], []
        for _ in range(batch_size):
            if (index + time_steps) >= (self.samples):
                batchx.append(xs[index : -1])
                batchy.append(ys[-1])
                self.batch_index = -1
                break
            batchx.append(xs[index : index + time_steps])
            batchy.append(ys[index + time_steps])
            index += 1
        self.batch_index = index
        
        if len(batchx) == 0: return None
        
        batchx = torch.tensor(np.array(batchx, dtype=np.float32))
        batchy = torch.tensor(np.array(batchy, dtype=np.float32))
        
        return batchx, batchy
        
        
    
    
    
    def get_batch(
            self,
            batch_size : int,
            time_steps : int = 32,
            include_only_last_y : bool = True,
            split : Literal['train', 'test', 'valid'] = 'train',
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
        
        The shape of `xs` is (batch_dim, time_window, height, width, channels).
        
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
        
        # batchx = np.transpose(batchx, (0, 4, 1, 2, 3))
        
        batchx = torch.tensor(batchx, dtype=torch.float32)
        batchy = torch.tensor(batchy, dtype=torch.float32)
        
        return (batchx, batchy)
    
    def to(self, device : Literal['cuda', 'cpu']):
        """Moves all the data in the dataset to the specified device as pytorch tensors.

        Args:
            device (Literal[&#39;cuda&#39;, &#39;cpu&#39;]): target device.
        """
        self.x_data = torch.tensor(self.x_data, device=device)
        self.y_vel = torch.tensor(self.y_vel, device=device)
        self.y_pos = torch.tensor(self.y_pos, device=device)
        
        return self


def load_dataset(path : str, quiet : bool = False) -> Dataset:
    """Loads a dataset from a `.rdata` dataset file."""
    dataset = Dataset()
    dataset.load(path, quiet = quiet)
    return dataset




if __name__ == '__main__':
    d = Dataset()
    d.load('ds1')
    
    
    
    

