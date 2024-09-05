import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.io as io
from torchvision.transforms.functional import resize, to_grayscale
import torch.nn.functional as f
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from skimage import measure
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AlteredMNIST(Dataset):
    def __init__(self):
        self.aug_directory = 'Data\\aug'
        self.clean_directory='Data\\clean'
        self.mapping=self.create_mapping()
        self.data = []
        self.clean_data = []
        self.labels=[]
        self.load_data()
        
    def create_mapping(self):
        mapping = defaultdict(list)
        
        clean_images = {}
        for clean_image in os.listdir(self.clean_directory):
            parts = clean_image.split('_')
            clean_label = parts[-1].split('.')[0]
            clean_images[clean_label] = os.path.join(self.clean_directory, clean_image)

        for augmented_image in os.listdir(self.aug_directory):
            parts = augmented_image.split('_')
            augmented_label = parts[-1].split('.')[0]

            if augmented_label in clean_images:
                clean_image = clean_images[augmented_label]
                mapping[augmented_image].append(clean_image) 
        
        return mapping

    def load_data(self):
        for filename in os.listdir(self.aug_directory):
            parts = filename.split('_')
            label = int(parts[2].split('.')[0])
            
            # Load the image data
            img_path = os.path.join(self.aug_directory, filename)
            img_data = io.read_image(img_path, mode=io.image.ImageReadMode.GRAY).float()
            img_data = resize(img_data, (28, 28))
            
            clean_img_path = self.mapping[filename][0]
            clean_img_data = io.read_image(clean_img_path, mode=io.image.ImageReadMode.GRAY).float()
            clean_img_data = resize(clean_img_data, (28, 28))

            # Append the data and label
            self.data.append(img_data)
            self.clean_data.append(clean_img_data)
            self.labels.append(label)

        self.data = torch.stack(self.data) 
        self.clean_data = torch.stack(self.clean_data)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.clean_data[index]

    def __len__(self):
        return len(self.data)
   
   
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)  
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  
            ResidualBlock(128, 256, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 256, 64)
        self.fc_mu = nn.Linear(64, 32)
        self.fc_logvar = nn.Linear(64, 32)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  
        out = self.bn2(out) 
        out = self.relu(out)
        out = self.res_blocks(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 7 * 7 * 256)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(128)  
        self.relu = nn.ReLU(inplace=True)  
        self.res_blocks = nn.Sequential(
            ResidualBlock(128, 64),
            ResidualBlock(64, 32)
        )
        self.conv_transpose1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample((28, 28))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.unflatten(out)
        out = self.conv1(out) 
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.res_blocks(out)
        out = self.conv_transpose1(out)
        out = self.upsample(out)
        return out


class AELossFn(nn.Module):
    def __init__(self):
        super(AELossFn, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, original, reconstructed):
        return self.loss_fn(original, reconstructed)


class VAELossFn(nn.Module):
    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        BCE = f.cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def ParameterSelector(E,D):
    return list(E.parameters()) + list(D.parameters())


class AETrainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu):
        device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.best_ssim = float('-inf')  
        self.train()
        
    def plot_tsne(self,embeddings, epoch):
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='r', marker='o')
        ax.set_title(f't-SNE Visualization at Epoch {epoch+1}')
        plt.savefig(f'AE_epoch_{epoch+1}.png')
        plt.close()

    def train(self):
        for epoch in range(50): 
            running_loss = 0.0
            running_loss_minibatch = 0.0
            running_ssim = 0.0
            running_ssim_minibatch = 0.0
            embeddings = []
            for i, data in enumerate(self.dataloader, 0):
                inputs, clean_inputs = data
                inputs = inputs
                clean_inputs = clean_inputs
                self.optimizer.zero_grad()
                
                encoded_inputs,_ = self.encoder(inputs)
                embeddings.append(encoded_inputs.detach().cpu().numpy())

                outputs = self.decoder(encoded_inputs)
                loss = self.loss_fn(inputs, outputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_loss_minibatch+=loss.item()

                for img,clean_img in zip(outputs,clean_inputs):
                    ssim=structure_similarity_index(img,clean_img)
                    tensor_ssim=torch.tensor([ssim])
                    if(torch.isnan(tensor_ssim)):
                        ssim=1
                    running_ssim+=ssim/outputs.shape[0]
                    running_ssim_minibatch+=ssim/outputs.shape[0]
                    

                if i % 10 == 9:   
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch+1, i+1, running_loss_minibatch/10, running_ssim_minibatch/10))
                    running_loss_minibatch = 0.0
                    running_ssim_minibatch = 0.0
                    
            avg_loss = running_loss/len(self.dataloader)
            avg_ssim = running_ssim/len(self.dataloader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch+1, avg_loss, avg_ssim))

            # Plot 3D t-SNE after every 10 epochs
            if epoch % 10 == 9:
                embeddings = np.concatenate(embeddings)
                self.plot_tsne(embeddings,epoch) 
                
            if avg_ssim > self.best_ssim:
                self.best_ssim = avg_ssim
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'ssim': avg_ssim,
                }, "AE_Model.pth")

        print('Finished Training')


class VAETrainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu):
        self.device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_ssim = float('-inf')  
        self.train()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def plot_tsne(self,embeddings, epoch):
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='b', marker='o')
        ax.set_title(f't-SNE Visualization at Epoch {epoch+1}')
        plt.savefig(f'VAE_epoch_{epoch+1}.png')
        plt.close()

    def train(self):
        for epoch in range(50):
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            running_loss_minibatch = 0.0
            running_ssim_minibatch = 0.0
            embeddings = []
            for i, data in enumerate(self.dataloader, 0):
                inputs, clean_inputs = data
                self.optimizer.zero_grad()

                mu, logvar = self.encoder(inputs)
                z = self.reparameterize(mu, logvar)
                embeddings.append(z.detach().cpu().numpy())
                recon_batch = self.decoder(z)
                loss = self.loss_fn(recon_batch, inputs, mu, logvar)
                loss.backward()
                self.optimizer.step()
            
            
                for img,clean_img in zip(recon_batch,clean_inputs):
                    ssim=structure_similarity_index(img,clean_img)
                    tensor_ssim=torch.tensor([ssim])
                    if(torch.isnan(tensor_ssim)):
                        ssim=1
                    running_ssim+=ssim/recon_batch.shape[0]
                    running_ssim_minibatch+=ssim/recon_batch.shape[0]
                    

                running_loss += loss.item()
                running_loss_minibatch+= loss.item()
                
                if i % 10 == 9:   
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity{}:".format(epoch + 1, i + 1, running_loss_minibatch / 10, running_ssim_minibatch/10))
                    running_loss_minibatch = 0.0
                    running_ssim_minibatch = 0.0
                    
            avg_loss = running_loss/len(self.dataloader)
            avg_ssim = running_ssim/len(self.dataloader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch + 1, avg_loss, avg_ssim))

            if epoch % 10 == 9:
                embeddings = np.concatenate(embeddings)
                self.plot_tsne(embeddings,epoch) 
                
            if avg_ssim > self.best_ssim:
                self.best_ssim = avg_ssim
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'ssim': avg_ssim,
                }, "VAE_Model.pth")

        print('Finished Training')


class AE_TRAINED:
    def __init__(self,gpu):
        pass
            
    @staticmethod
    def from_path(sample, original, type):
        encoder = Encoder()
        decoder = Decoder()
        
        checkpoint = torch.load('AE_model.pth')

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        
        "Compute similarity score of both 'sample' and 'original' and return in float"
        img_data = io.read_image(sample, mode=io.image.ImageReadMode.GRAY).float()
        img_data = resize(img_data, (28, 28))
        img_data = img_data.unsqueeze(0)
        
        org_img_data = io.read_image(original, mode=io.image.ImageReadMode.GRAY).float()
        org_img_data = resize(org_img_data, (28, 28))
        org_img_data = org_img_data.unsqueeze(0)
        
        with torch.no_grad():
            encoded_sample, _ = encoder(img_data)
            decoded_sample = decoder(encoded_sample)

        if type == 'SSIM':
            ssim=structure_similarity_index(decoded_sample,org_img_data)
            tensor_ssim=torch.tensor([ssim])
            if(torch.isnan(tensor_ssim)):
                ssim=1
            return ssim
        elif type == 'PSNR':
            psnr = peak_signal_to_noise_ratio(decoded_sample,org_img_data)
            return psnr
            
            
class VAE_TRAINED:
    def __init__(self,gpu):
        pass
            
    @staticmethod
    def from_path(sample, original, type):
        encoder = Encoder()
        decoder = Decoder()
        
        checkpoint = torch.load('AE_model.pth')

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        "Compute similarity score of both 'sample' and 'original' and return in float"
        img_data = io.read_image(sample, mode=io.image.ImageReadMode.GRAY).float()
        img_data = resize(img_data, (28, 28))
        img_data = img_data.unsqueeze(0)
        
        org_img_data = io.read_image(original, mode=io.image.ImageReadMode.GRAY).float()
        org_img_data = resize(org_img_data, (28, 28))
        org_img_data = org_img_data.unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = encoder(img_data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
            recon_batch = decoder(z)

        if type == 'SSIM':
            ssim=structure_similarity_index(recon_batch,org_img_data)
            tensor_ssim=torch.tensor([ssim])
            if(torch.isnan(tensor_ssim)):
                ssim=1
            return ssim
        elif type == 'PSNR':
            psnr = peak_signal_to_noise_ratio(recon_batch,org_img_data)
            return psnr


class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass


class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,accuracy))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass


class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()


def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    
    mu1 = f.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = f.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = f.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = f.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  f.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()
