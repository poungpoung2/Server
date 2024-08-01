# Pytorch on Servers

## Installation
Create a venv first. Check the other markdown file that explains installing how to create a venv
`pip install --no-index torch==<version>` # Install the specified pytorch version
**Extra**
Also it is recommended to install the following libraries 
`pip install --no-index torch torchvision torchtext torchaudio`

## Job submission
pytorch-test.sh
```
#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out #Node_Name-Job_id.out
module load python/<version>
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index

python <script>.py
```
Submit the bash script
`sbatch pytorch-test.sh`

## PyTorch with multiple CPUs
With small scale models, using **multiple CPUS instead of using a GPU** is recommneded.

**Examples**
pytorch-multi-cpu.sh
```
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=<NUM>

#SBATCH --mem=16G
#SBATCH --time=0-05:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<account>

module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index

echo "Start training.."
time python test.py
```
For the test.py 
`torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))`

## Pytorch with Data Parallelism

### <a href="https://pytorch.org/docs/master/notes/ddp.html">DistributedDataParallel </a>

**Structures**
1. Process Group: Before DDP works, the **"Process Group"** should be created to allow communication amog the GPUs
2. Model Syncing: For first initializing, the DDP picks one machine as a *rank 0* as a basis and share it among the others.
3. Reducers and Buckets: Each machine creates a "Reducer" that ensures gradients are shared. And this is done by grouping gradients in **"buckets."**
4. Forward Pass: DDP performs forward pass like a normal model.  If `find_unsued_parameters=True`, it tracers the autograd graph from model output to find out parameters that are involved and only focus on thsoe for backward pass.
5. Backward Pass: DDP uses **"hooks" (small pieces of code)** to synchronize theg gradients accross al machines. When all buckets are ready, the `Reducer` calculates the mean of gradients.
6. Optimization step: Optimize local model in eahc machine as all the model replicas are in sync.

**Model for Parallel**
```
class EfficientNetModelParallel(nn.Module):
    def __init__(self, dev0, dev1):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        model = efficientnet_b0(pretrained=True)  # Load the EfficientNet-B0 model with pre-trained weights

        # Split the model into two parts and place them on different devices
        self.features0 = nn.Sequential(
            model.features[:4].to(dev0)  # The first 4 layers of the model on dev0 (first GPU)
        )
        self.features1 = nn.Sequential(
            model.features[4:].to(dev1)  # The remaining layers on dev1 (second GPU)
        )
        self.avgpool = model.avgpool.to(dev1)  # Place the average pooling layer on dev1
        self.classifier = model.classifier.to(dev1)  # Place the classifier on dev1

    def forward(self, x):
        x = x.to(self.dev0)  # Move input to dev0
        x = self.features0(x)  # Process input through the first part of the model on dev0
        x = x.to(self.dev1)  # Move intermediate output to dev1
        x = self.features1(x)  # Process through the second part of the model on dev1
        x = self.avgpool(x)  # Apply average pooling on dev1
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.classifier(x)  # Pass through the classifier
        return x  # Return the final output

```



**Script**
```
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Setup the distributed process group with rank and world size
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    # Clean up the distributed process group
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)  # Initialize process group
    model = ToyModel().to(rank)  # Initialize the model and move it to the device corresponding to the rank
    ddp_model = DDP(model, device_ids=[rank])  # Wrap the model with DDP

    loss_fn = nn.MSELoss()  # Define the loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # Define the optimizer

    optimizer.zero_grad()  # Zero the gradients
    output = ddp_model(torch.randn(20, 10))  # Forward pass
    label = torch.randn(20, 5).to(rank)  # Generate some random labels
    loss_fn(output, label).backward()  # Compute the loss and perform backpropagation
    optimizer.step()  # Update the model parameters

    cleanup()  # Clean up the process group

def run_demo(demo_fn, world_size):
    # Run the demo function across multiple processes
    mp.spawn(
        demo_fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)  # Initialize process group

    # Model Initialization
    model = ToyModel().to(rank)  # Initialize the model on the corresponding rank
    ddp_model = DDP(model, device_ids=[rank])  # Wrap the model with DDP

    # Create checkpoint path
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"  # Temporary file for saving checkpoint

    # Save the checkpoint
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    dist.barrier()  # Synchronize all processes

    loss_fn = nn.MSELoss()  # Define the loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)  # Define the optimizer

    optimizer.zero_grad()  # Zero the gradients
    output = ddp_model(torch.randn(20, 10))  # Forward pass
    label = torch.randn(20, 5).to(rank)  # Generate some random labels
    loss_fn(output, label).backward()  # Compute the loss and perform backpropagation
    optimizer.step()  # Update the model parameters

    if rank == 0:
        os.remove(CHECKPOINT_PATH)  # Remove the checkpoint file
    cleanup()  # Clean up the process group

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)  # Initialize process group
    dev0 = rank * 2  # Assign first GPU
    dev1 = rank * 2 + 1  # Assign second GPU
    model = EfficientNetModelParallel(dev0, dev1)  # Initialize the parallel model
    ddp_model = DDP(model)  # Wrap the model with DDP WITHOUT setting device_ids
    
    loss_fn = nn.MSELoss()  # Define the loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # Define the optimizer
    optimizer.zero_grad()  # Zero the gradients
    # outputs will be on dev1
    outputs = ddp_model(torch.randn(20, 10))  # Forward pass
    labels = torch.randn(20, 5).to(dev1)  # Generate some random labels on dev1
    loss_fn(outputs, labels).backward()  # Compute the loss and perform backpropagation
    optimizer.step()  # Update the model parameters

    cleanup()  # Clean up the process group

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    assert n_gpus >= 2, "Don't have enough GPUs"  # Ensure at least 2 GPUs are available
    world_size = n_gpus  # Set the world size to the number of GPUs
    run_demo(demo_basic, world_size)  # Run the basic demo
    run_demo(demo_checkpoint, world_size)  # Run the checkpoint demo
    world_size = n_gpus // 2  # Halve the world size for model parallel demo
    run_demo(demo_model_parallel, world_size)  # Run the model parallel demo
```

**Using torch elastic**
```
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)  # First linear layer
        self.relu = nn.ReLU()  # ReLU activation
        self.net2 = nn.Linear(10, 5)  # Second linear layer

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))  # Forward pass through both layers with ReLU activation

def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # Initialize process group
    print(f"Running DDP example on rank {rank}.")

    device_id = rank  # Set device id to rank
    model = ToyModel().to(rank)  # Initialize the model and move it to the corresponding GPU
    ddp_model = DDP(model, device_ids=[device_id])  # Wrap the model with DDP

    loss_fn = nn.MSELoss()  # Define the loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # Define the optimizer

    optimizer.zero_grad()  # Zero the gradients
    outputs = ddp_model(torch.randn(20, 10)).to(device_id)  # Forward pass on the corresponding device
    labels = torch.randn(20, 5).to(device_id)  # Generate some random labels
    loss_fn(outputs, labels).backward()  # Compute the loss and perform backpropagation
    optimizer.step()  # Update the model parameters

if __name__ == "__main__":
    spawn(train, args=(torch.cuda.device_count(),), nprocs=torch.cuda.device_count())  # Run the training process across all available GPUs		
```
	

	





