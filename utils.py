import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_cVAE_Hybrid(model, color_model, BIC_model, dataset, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    model.eval()
    color_model.eval()
    BIC_model.eval()

    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(
            DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # inferenc using vae_GSNN model and predict using forward model
        x_pred, mu, logvar, temp = model.inference(y)
        y_color_pred = color_model(x_pred, None)
        y_BIC_pred = BIC_model(x_pred, None)
        y_pred = torch.cat((y_color_pred, y_BIC_pred), dim=1)


        x_pred_raw = x_pred * range_[:x_dim] + min_[:x_dim]
        x_raw = x * range_[:x_dim] + min_[:x_dim]

        y_pred_raw = y_pred * range_[x_dim:] + min_[x_dim:]
        y_raw = y * range_[x_dim:] + min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())

        if show == 1:
            print("VAE net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('VAE Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()

def evaluate_forward(forward_model, dataset, show=0):
    # for evaluate the dataset itself.
    # x: structure ; y: CIE 
    # return: predicted CIE
    forward_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)

        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]
        M = x.size()[0]
        y_pred = forward_model(x, None)
        y_pred_raw = y_pred *range_[x_dim:] + min_[x_dim:]
        y_raw = y *range_[x_dim:] + min_[x_dim:]

        # get MSE for the design
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        
        if show==1:
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))
        
    y_pred_raw = y_pred_raw.cpu().numpy()
    y_raw = y_raw.cpu().numpy()
    return   y_raw, y_pred_raw

def evaluate_tandem(inverse_model, color_model, BIC_model, dataset, show=1):
    # x: structure. y: color and BIC
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    inverse_model.eval()
    color_model.eval()
    BIC_model.eval()

    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(
            DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # inferenc using vae_GSNN model and predict using forward model
        x_pred = inverse_model.pred(y)
        y_color_pred = color_model(x_pred, None)
        y_BIC_pred = BIC_model(x_pred, None)
        y_pred = torch.cat((y_color_pred, y_BIC_pred), dim=1)


        x_pred_raw = x_pred * range_[:x_dim] + min_[:x_dim]
        x_raw = x * range_[:x_dim] + min_[:x_dim]

        y_pred_raw = y_pred * range_[x_dim:] + min_[x_dim:]
        y_raw = y * range_[x_dim:] + min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())

        if show == 1:
            print("VAE net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('VAE Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()


def evaluate_vae(model,color_model,BIC_model,dataset, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    model.eval()
    color_model.eval()
    BIC_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(
            DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # inferenc using vae_GSNN model and predict using forward model
        x_pred, mu, logvar, temp = model.inference(y)
        y_color_pred = color_model(x_pred, None)
        y_BIC_pred = BIC_model(x_pred, None)
        y_pred = torch.cat((y_color_pred, y_BIC_pred), dim=1)

        x_pred_raw = x_pred * range_[:x_dim] + min_[:x_dim]
        x_raw = x * range_[:x_dim] + min_[:x_dim]

        y_pred_raw = y_pred * range_[x_dim:] + min_[x_dim:]
        y_raw = y * range_[x_dim:] + min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())

        if show == 1:
            print("VAE net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('VAE Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()

def evaluate_gan(gan_model, forward_color_model,forward_BIC_model, dataset, show=1):
    # evaluate both the gan model using a forward model
    # y: structure. x: CIE

    '''
    returns:
        x_raw: original desired xyY
        x_raw_pred: xyY predicted by the forward module for the inversely designed structure
        y_raw: original structure parameters
        y_raw_pred: inversely designed parameters.
    '''

    gan_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        z = gan_model.sample_noise(len(x)).to(DEVICE)
        y_pred = gan_model.generator(x, z)
        x_color_pred = forward_color_model(y_pred, None)
        x_BIC_pred = forward_BIC_model(y_pred, None)
        x_pred = torch.cat((x_color_pred, x_BIC_pred), dim=1)

        x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        x_raw =  x *range_[:x_dim] +min_[:x_dim] 
        
        y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
        y_raw =  y *range_[x_dim:] +min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())

        if show==1:
            print("GAN net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('GAN Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()
