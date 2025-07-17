import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Linear(256, 64),
           # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.ReLU()
        )
    def forward(self, x, y):
        return self.layers(x)

class MLP_color(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP_color, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        return self.layers(x)


class MLP_BIC(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP_BIC, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x, y):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




class cVAE_hybrid(nn.Module):

    def __init__(self, color_model,BIC_model, vae_model):
        super(cVAE_hybrid, self).__init__()
        self.color_model = color_model
        self.BIC_model = BIC_model
        self.vae_model = vae_model

    def forward(self, x, y):
        # the prediction is based on cVAE_GSNN model
        '''
        Pass the desired target x to the vae_hybrid network.
        '''

        x_pred, mu, logvar, x_hat = self.vae_model(x, y)

        y_color_pred = self.color_model(x_pred, None)
        y_BIC_pred = self.BIC_model(x_pred, None)
        y_pred = torch.cat((y_color_pred,y_BIC_pred),dim=1)
        return x_pred, mu, logvar, x_hat, y_pred

    def pred(self, x):
        y_color_pred = self.color_model(x, None)
        y_BIC_pred = self.BIC_model(x, None)
        pred = torch.cat((y_color_pred,y_BIC_pred),dim=1)
        return pred




class cVAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=4):
        super(cVAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        y_pred = y
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, y_pred

    def inference(self, y):
        mu, logvar = torch.zeros([y.size()[0], self.latent_dim]), torch.zeros([y.size()[0], self.latent_dim])
        mu = mu.cuda()
        logvar = logvar.cuda()
        z = self.reparameterize(mu, logvar)
        print(mu.device,logvar.device,z.device,y.device,type(z))
        # z = z.cuda()
        return self.decode(z, y), mu, logvar, y

class TandemNet(nn.Module):

    def __init__(self, color_model,BIC_model, inverse_model):
        super(TandemNet, self).__init__()
        self.color_model = color_model
        self.BIC_model = BIC_model
        self.inverse_model = inverse_model

    def forward(self, x, y):
        # x: structure, y: color, BIC
        '''
        Pass the desired target x to the tandem network.
        '''

        pred = self.inverse_model(y, x)
        y_color_pred = self.color_model(pred, None)
        y_BIC_pred = self.BIC_model(pred, None)

        return torch.cat((y_color_pred,y_BIC_pred),dim=1)

    def pred(self, y):
        pred = self.inverse_model(y, None)
        # pred : structure
        return pred

class TandemNet_inverse(nn.Module):

    def __init__(self, input_size, output_size):
        super(TandemNet_inverse, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.ReLU()
        )
    def forward(self, x, y):
        return self.layers(x)
    

class Generator(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(Generator, self).__init__()

        self.input_size = input_size

        self.net = nn.Sequential(*[nn.Linear(input_size + noise_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, noise):
        y = self.net(torch.cat((x, noise), dim=1))
        return y

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(*[nn.Linear(input_size+output_size, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   #nn.BatchNorm1d(hidden_dim), #
                                   #don't use batch norm for the D input layer and G output layer to aviod the oscillation and model instability 
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2)])
        

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128, 3))

    def forward(self, y_fake, x):
        h = self.net(torch.cat((y_fake, x), dim=1))
        validity = self.adv_layer(h)
        # label = self.aux_layer(h)

        return validity

class cGAN(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(cGAN, self).__init__()

        self.generator = Generator(
            input_size, output_size, noise_dim=noise_dim, hidden_dim=hidden_dim)
        self.discriminator = Discriminator(
            output_size, input_size, hidden_dim=hidden_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):

        y_fake = self.generator(x, noise)
        validity = self.discriminator(y_fake, x)

        return validity

    def sample_noise(self, batch_size, prior=1):

        if prior == 1:
            z = torch.tensor(np.random.normal(0, 1, (batch_size, self.noise_dim))).float()
        else:
            z = torch.tensor(np.random.uniform(0, 1, (batch_size, self.noise_dim))).float()
        return z

    def sample_noise_M(self, batch_size):
        M = 100
        z = torch.tensor(np.random.normal(
            0, 1, (batch_size*M, self.noise_dim))).float()
        return z