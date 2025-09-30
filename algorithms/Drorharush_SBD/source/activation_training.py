import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model_tools import ActivationLoss
from dataset import QPIDataSet
from model import ActivationNet, LISTA
from torch.optim import Adam
from tqdm import tqdm
import os
from pathlib import Path
from matplotlib import pyplot as plt


def plot_result(val_vs_epoch, training_vs_epoch, folder=None):
    plt.plot(val_vs_epoch, label='Validation', alpha=0.5)
    plt.plot(training_vs_epoch, label='Training', alpha=0.5)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch number')
    plt.tight_layout()
    fig_path = 'loss_vs_epoch' if folder is None else f'{folder}/loss_vs_epoch'
    plt.savefig(fig_path)
    plt.show()


def plot_example(dataset, network, idx, folder=None):
    meas, ker, act = dataset[idx]

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    fig.suptitle(f'Prediction {idx}')
    ax[0].set_title('Input', fontsize=12)
    ax[0].imshow(meas[0].cpu(), cmap='hot')

    ax[1].set_title('Network output', fontsize=12)
    network.eval()
    network.cpu()
    predicted = network(meas.unsqueeze(1).cpu())[0][0].data.numpy()
    ax[1].imshow(predicted, cmap='hot')

    ax[2].set_title('Target', fontsize=12)
    ax[2].imshow(act.cpu(), cmap='hot')

    for i in range(3):
        ax[i].set_axis_off()
    path_name = Path(f'Pred{idx}') if folder is None else Path(f'{folder}/Pred{idx}')
    plt.savefig(path_name)


def compute_loss(dataloader, network, loss_function):
    loss = 0

    if torch.cuda.is_available():
        network.cuda()
    network.eval()

    n_batches = 0
    with torch.no_grad():
        for mes, ker, act in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                mes = mes.cuda()
                ker = ker.cuda()
                act = act.cuda()
            act_pred = network(mes)

            loss += loss_function(act_pred, act, ker, mes)

    loss /= n_batches
    return loss


print('Loading QPI datasets.')
batch_size = 20
train_ds = QPIDataSet(f'{os.getcwd()}/training_dataset')
valid_ds = QPIDataSet(f'{os.getcwd()}/validation_dataset')
training_dataloader = DataLoader(train_ds, batch_size=batch_size)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size)

# Defining the network, optimizer and loss function.
layer_number = 5
net = LISTA(layer_number)
# net = ActivationNet()
optimizer = Adam(net.parameters(), lr=1e-4)  # Increased from 1e-4 to 1e-3, maybe not a good idea
scheduler = ReduceLROnPlateau(optimizer)  # Will reduce learning rate after 10 epochs with no improvement
sparsity = 0.01
loss_func = ActivationLoss(sparsity)

#  Getting parameters from my last model, and loading loss.
trained_model_path = Path(f'trained_lista_{layer_number}layers.pt', map_location=torch.device('cpu'))
val_loss_vs_epoch_path = Path(f'val_vs_loss_{layer_number}layers.npy')
training_loss_vs_epoch_path = Path(f'training_vs_loss_{layer_number}layers.npy')

validation_loss_vs_epoch = np.load(val_loss_vs_epoch_path).tolist() if val_loss_vs_epoch_path.is_file() else []
training_loss_vs_epoch = np.load(training_loss_vs_epoch_path).tolist() if training_loss_vs_epoch_path.is_file() else []


if trained_model_path.is_file():
    net.load_state_dict(torch.load(trained_model_path))
    print('Loading parameters from your last model.')
    net.eval()

# Training loop:

folder = f'lista{layer_number}layers'
n_epochs = 100

if torch.cuda.is_available():
    net.cuda()
    print('Using GPU.')
else:
    print('Using CPU.')

pbar = tqdm(range(n_epochs))

for epoch in pbar:

    if len(validation_loss_vs_epoch) > 1:
        rounded_loss = np.format_float_scientific(validation_loss_vs_epoch[-1], exp_digits=2, precision=4)
        rounded_best = np.format_float_scientific(np.min(validation_loss_vs_epoch), exp_digits=2, precision=4)
        arg_best = np.argmin(validation_loss_vs_epoch)
        pbar.set_description(f'epoch: {epoch}.'
                             f' val loss({len(validation_loss_vs_epoch)}): {rounded_loss}.'
                             f' best({arg_best + 1}): {rounded_best}')

    net.train()  # put the net into "training mode"
    for measurement, kernel, activation in training_dataloader:
        if torch.cuda.is_available():
            measurement = measurement.cuda()
            kernel = kernel.cuda()
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        pred_activation = net(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)

        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode

    valid_loss = compute_loss(valid_dataloader, net, loss_func)
    scheduler.step(valid_loss)
    training_loss = compute_loss(training_dataloader, net, loss_func)
    validation_loss_vs_epoch.append(valid_loss.cpu().detach().numpy())
    training_loss_vs_epoch.append(training_loss.cpu().detach().numpy())

    # Saving model parameters only if the validation loss is minimal
    if len(validation_loss_vs_epoch) == 1 or validation_loss_vs_epoch[-1] == min(validation_loss_vs_epoch):
        torch.save(net.state_dict(), trained_model_path)
    np.save(val_loss_vs_epoch_path, validation_loss_vs_epoch)
    np.save(training_loss_vs_epoch_path, training_loss_vs_epoch)
    # if there is no improvement in the last fraction of the epochs, stop training.
    no_improvement_ratio = (len(validation_loss_vs_epoch) - np.argmin(validation_loss_vs_epoch)) / len(validation_loss_vs_epoch)
    if epoch > 100 and no_improvement_ratio > 1/3:
        print('No improvement seen in the last third. Stopped training.')
        break
    plot_result(validation_loss_vs_epoch, training_loss_vs_epoch, folder=folder)


# Plotting results
plot_result(validation_loss_vs_epoch, training_loss_vs_epoch, folder=folder)

# Plot one example for reference
idxs = np.random.choice(valid_ds.length, 10, replace=False)
for i in idxs:
    plot_example(valid_ds, net, i, folder=folder)
