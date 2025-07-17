from utils import count_params

from datasets import get_dataloaders
from torch.optim.lr_scheduler import StepLR
from models import TandemNet,MLP_BIC,MLP_color,TandemNet_inverse

import torch
from torch import nn
import numpy as np

import argparse
import wandb

seed = 418
torch.manual_seed(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# For cVAE or cVAE_hybrid model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python cvae.py

def train(model, train_loader, optimizer, criterion, configs):
    # x: structure ; y: CIE

    model.inverse_model.train()
    model.color_model.eval()
    model.BIC_model.eval()

    loss_epoch = 0

    for x, y in train_loader:

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        y_pred = model(x, y)

        loss = criterion(y_pred, y)  


        loss.backward()
        optimizer.step()

        loss_epoch += loss * len(x)

    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch


def evaluate(model, val_loader, test_loader, optimizer, criterion, configs, test=False):
    # x: structure ; y: CIE


    model.eval()

    dataloader = test_loader if test else val_loader

    with torch.no_grad():
        loss_epoch = 0
        loss_pred = 0

        for x, y in dataloader:

            x, y = x.to(DEVICE), y.to(DEVICE)

            y_pred = model(x, y)

            loss = criterion(y_pred, y) 

            loss_epoch += loss * len(x)
            loss_pred += loss * len(x)

        loss_epoch = loss_epoch / len(dataloader.dataset)
        loss_pred = loss_pred / len(dataloader.dataset)

    return loss_epoch, loss_pred


def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_all': loss_all,
        'configs': {key: str(value) for key, value in configs.items()},
        'seed': seed,
    }, path)


def main(configs):

    # Initialize wandb
    wandb.login(key="a2481ca949472cf0a93ab773c7b80f2c01001f0e")
    wandb.init(project="VAE-Forward-Color",name=configs.name,notes=configs.notes, config=configs)
    config = wandb.config

    train_loader, val_loader, test_loader = get_dataloaders('vae_hybrid', config.batch_size)

    color_model = MLP_color(6, 3).to(DEVICE)
    color_model.load_state_dict(torch.load(config.save_path + "/" + config.color_model + ".pth")['model_state_dict'])
    BIC_model = MLP_BIC(6, 1).to(DEVICE)
    BIC_model.load_state_dict(torch.load(config.save_path + "/" + config.BIC_model + ".pth")['model_state_dict'])

    inverse_model = TandemNet_inverse(config.output_dim, config.input_dim).to(DEVICE)
    model = TandemNet(color_model,BIC_model, inverse_model)

    # set up optimizer and criterion

    # optimizer = torch.optim.Adam(model.-.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    optimizer = torch.optim.Adam(model.inverse_model.parameters(), lr=config.lr)

    scheduler = StepLR(optimizer, step_size=config.epoch_lr_de, gamma=config.lr_de)

    criterion = nn.MSELoss()

    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))

    # start training
    # start training
    path = config.save_path + "/" + config.name + ".pth"
    path_temp = config.save_path + "/" + config.name + "_temp.pth"    
    epochs = config.epochs
    loss_all = np.zeros([3, config.epochs])
    loss_val_best = 100

    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion, config)
        loss_train, temp = evaluate(model, train_loader, test_loader, optimizer, criterion, config)
        loss_val, loss_pred = evaluate(model, val_loader, test_loader, optimizer, criterion, config)
        loss_all[0, e] = loss_train
        loss_all[1, e] = loss_val
        loss_all[2, e] = loss_pred

        if loss_val_best >= loss_all[1, e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1, e]
            save_checkpoint(model, optimizer, e, loss_all, path, config)

        if config.if_lr_de:
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            lr = config.lr

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, pred loss {:.6f},  lr {:.6f}, best {:.6f}.'.format(e,
                                                                                                                loss_train,
                                                                                                                loss_val,
                                                                                                                loss_pred,
                                                                                                                lr,
                                                                                                                loss_val_best))

        if e % 10 == 0:
            save_checkpoint(model, optimizer, e, loss_all, path_temp, config)

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='cVAE_GSNN')
    parser.add_argument('--input_dim', type=int, default=6, help='Input dimension of structure')
    parser.add_argument('--output_dim', type=int, default=4, help='Output dim of color and BIC')

    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of latent variable')
    parser.add_argument('--add_forward', action='store_true', default='True',
                        help='Connect cVAE to forward model or not')
    parser.add_argument('--weight_forward', type=float, default=10.0,
                        help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for forward model')
    parser.add_argument('--if_lr_de', action='store_true', default='True',
                        help='If decrease learning rate duing training')  # to enable step lr, add argument '--if_lr_de'
    parser.add_argument('--lr_de', type=float, default=0.2, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=2000, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization')
    parser.add_argument('--save_path', type=str, default="./model", help='Run multiple times and save ')
    parser.add_argument('--color_model', type=str, default="forward_04", help='Run multiple times and save ')
    parser.add_argument('--BIC_model', type=str, default="forwardBIC_02", help='Run multiple times and save ')
    parser.add_argument('--name', type=str, default="tandem_01", help='Run multiple times and save ')
    parser.add_argument('--notes', type=str, default="tandem第1次尝试", help='Run multiple times and save ')
  
    args = parser.parse_args()

    main(args)