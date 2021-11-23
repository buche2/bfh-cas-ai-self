import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import os

from MyDataset import MyDataset
from VAE import VAE

class Trainer():

    def __init__ (self):
        super(Trainer, self).__init__()

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        torch.manual_seed(0)

        dataset = MyDataset()
        train_loader, test_loader, val_loader = dataset.get_loaders()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_obs = dataset.column_count
        print(self.n_obs)
        self.model = VAE(self.n_obs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data[0].to(self.device)

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def save(self):
        path = './runs/{}/'.format('vae')
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.model.state_dict(), os.path.join(path, 'vae_state_dict'))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.6f}'.format(test_loss))

        # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x.view(-1, 489), reduction='mean')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD