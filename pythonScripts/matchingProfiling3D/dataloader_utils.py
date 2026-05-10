import os
from easydict import EasyDict as edict
from predator.datasets.dataloader import get_dataloader, get_datasets, collate_fn_descriptor
from predator.lib.utils import load_config

class PredatorDataLoader:
    def __init__(self, config_path, split='train'):
        """
        Predator DataLoader wrapper for profiling scripts.

        Args:
            config_path (str): Path to the predator config yaml.
            split (str): Dataset split to load ('train' or 'val').
        """
        # Load Predator config
        self.config = load_config(config_path)
        self.config['snapshot_dir'] = '%s' % self.config['exp_dir']
        self.config['tboard_dir'] = '%s/tensorboard' % self.config['exp_dir']
        self.config['save_dir'] = '%s/checkpoints' % self.config['exp_dir']
        self.config = edict(self.config)

        # Define architectures
        architectures = {
            'indoor': [
                'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
                'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
                'resnetb', 'resnetb', 'nearest_upsample', 'unary',
                'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary'
            ]
        }
        self.config.architecture = architectures[self.config.dataset]

        # Get datasets
        train_set, val_set, benchmark_set = get_datasets(self.config)

        if split == "train":
            self.dataset = train_set
        elif split == "val":
            self.dataset = val_set
        else:
            raise ValueError(f"Unknown data type: {split}")

        self.dataset_size = len(self.dataset)

        # Initialize dataloader
        self.train_loader, self.neighborhood_limits = get_dataloader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        self.data_iter = iter(self.train_loader)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """Direct indexed access to a raw sample (returns tuple, not collated)."""
        if index < 0 or index >= self.dataset_size:
            raise IndexError(f"Index {index} out of range [0, {self.dataset_size})")
        return self.dataset[index]

    def get_by_index(self, index):
        """Direct indexed access with collation applied (matches DataLoader output format)."""
        if index < 0 or index >= self.dataset_size:
            raise IndexError(f"Index {index} out of range [0, {self.dataset_size})")
        raw_sample = self.dataset[index]
        return collate_fn_descriptor([raw_sample], self.config, self.neighborhood_limits)

    def get_next(self):
        """Returns the next batch from the dataloader (sequential iteration)."""
        return next(self.data_iter)
