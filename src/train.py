import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from helpers import after_subplot
import torch.nn as nn
criterion = nn.CrossEntropyLoss()


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Train the model for one epoch.

    Args:
        train_dataloader: The dataloader for the training data.
        model: The model to be trained.
        optimizer: The optimizer used for updating the model parameters.
        loss: The loss function used for calculating the loss.

    Returns:
        The average training loss for the epoch.
    """

    if torch.cuda.is_available():
        # transfer the model to the GPU
        model = model.cuda()

    # set the model to training mode
    model.train()
    
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,):

        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data)
        
        # 3. calculate the loss
        loss_value  = criterion(output, target)
        
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()
        
        # 5. perform a single optimization step (parameter update)
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss))

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
        Calculate the validation loss for one epoch.

        Parameters:
            valid_dataloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
            model (torch.nn.Module): The trained model.
            loss (torch.nn.Module): The loss function.

        Returns:
            float: The average validation loss for the epoch.
    """


    with torch.no_grad():

        # set the model to evaluation mode
        
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,):
            
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data)
            
            # 2. calculate the loss
            loss_value  = criterion(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss))

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    """
    Optimizes a model using the specified data loaders, model, optimizer, loss function, and training parameters.

    Parameters:
    - data_loaders: A dictionary containing the data loaders for training and validation datasets.
    - model: The model to be optimized.
    - optimizer: The optimizer used for updating the model's parameters.
    - loss: The loss function used for calculating the loss.
    - n_epochs: The number of epochs for training.
    - save_path: The path to save the optimized model.
    - interactive_tracking: (optional) A boolean flag indicating whether to enable interactive tracking of the training progress.

    Returns:
    None
    """

    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a plateau
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, verbose=True, threshold=0.01, min_lr=1e-6, patience=5)
    
#     scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True
# )

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss)

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print("Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss))

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01):
            
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path 
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    """
    Perform one epoch of testing on the provided test dataloader using the trained model and loss function.

    Parameters:
    - test_dataloader (torch.utils.data.DataLoader): The dataloader containing the test data.
    - model (torch.nn.Module): The trained model to be evaluated.
    - loss (torch.nn.Module): The loss function used to calculate the test loss.

    Returns:
    - test_loss (float): The average test loss over the entire epoch.
    """

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(enumerate(test_dataloader),
                                            desc='Testing',
                                            total=len(test_dataloader),
                                            leave=True,
                                            ncols=80):
            
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data)
            
            # 2. calculate the loss
            loss_value  = criterion(logits, target)

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred  = torch.argmax(logits, dim=1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=32, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from optimization import get_optimizer, get_loss
    from model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
