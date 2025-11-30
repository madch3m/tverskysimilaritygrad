#!/usr/bin/env python3

"""

Multi-GPU training launcher using torch.multiprocessing.

"""



import torch

import torch.multiprocessing as mp

from .optimized_trainer import DistributedTrainer





def launch_distributed_training(

    model_fn,

    train_dataset,

    test_dataset,

    world_size: int = 4,

    num_epochs: int = 10,

    batch_size: int = 256,

    learning_rate: float = 0.001

):

    """

    Launch distributed training across multiple GPUs.

    

    Args:

        model_fn: Function that returns a new model instance

        train_dataset: Training dataset

        test_dataset: Test/validation dataset

        world_size: Number of GPUs to use

        num_epochs: Number of training epochs

        batch_size: Batch size per GPU

        learning_rate: Learning rate

    """

    

    def train_worker(rank, world_size, model_fn, train_dataset, test_dataset, num_epochs, batch_size, lr):

        """Worker function for each GPU."""

        model = model_fn()

        

        trainer = DistributedTrainer(

            rank=rank,

            world_size=world_size,

            model=model,

            num_epochs=num_epochs,

            learning_rate=lr

        )

        

        trainer.train(train_dataset, test_dataset, batch_size=batch_size)

    

    mp.spawn(

        train_worker,

        args=(world_size, model_fn, train_dataset, test_dataset, num_epochs, batch_size, learning_rate),

        nprocs=world_size,

        join=True

    )

