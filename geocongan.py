# loading in and transforming data
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image

import torch.optim as optim

# import save code
from helpers import save_samples, checkpoint

# import models and utils
from cyclegan import Discriminator, CycleGenerator
from silnet import SilNet
from utils_cyclegan import imshow, scale, print_models, view_samples
from utils_silnet import reverse_transform, masks_to_colorimg
#from loss import silnet_loss#, cycle_consistency_loss
from hands_dataset import HandsDataset

from tqdm import tqdm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/((self.n_epochs - self.decay_start_epoch) + 1)

def get_data_loader(image_type, image_dir='synth2real',
                    image_size=256, batch_size=8, num_workers=0):
    """Returns training and test data loaders for a given image type, either 'synth' or 'real'.
    """

    # get training and test directories
    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = HandsDataset(image_type, train_path)
    test_dataset = HandsDataset(image_type, test_path)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    return train_loader, test_loader

def create_model(n_res_blocks=9, device="cpu"):
    """Builds the generators and discriminators."""

    # Instantiate generators
    G_XtoY = CycleGenerator(n_residual_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(n_residual_blocks=n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator()
    D_Y = Discriminator()

    # move models to GPU, if available
    if device != "cpu":
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_iters=200):

    print_every = 1#n_epochs//10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X, mask_fixed_X = test_iter_X.next()
    fixed_Y, mask_fixed_Y = test_iter_Y.next()

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    n_batches = min(len(iter_X), len(iter_Y))

    """for param in S.parameters():
        param.requires_grad = False"""

    for it in tqdm(range(1, n_iters+1), desc="Iteration"):

        if it % n_batches == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, silhouette_X = next(iter_X)
        #images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, silhouette_Y = next(iter_Y)
        #images_Y = scale(images_Y)

        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)
        silhouette_X = silhouette_X.to(device)
        silhouette_Y = silhouette_Y.to(device)


        # FORWARD PASS

        fake_X = G_YtoX(images_Y)
        fake_Y = G_XtoY(images_X)
        """with torch.no_grad():
            sil_fake_X = torch.sigmoid(S(fake_X))
            sil_fake_Y = torch.sigmoid(S(fake_Y))"""

        reconstructed_Y = G_XtoY(fake_X)
        reconstructed_X = G_YtoX(fake_Y)

        for param in D_X.parameters():
            param.requires_grad = False
        for param in D_Y.parameters():
            param.requires_grad = False

        g_optimizer.zero_grad()

        out_x = D_X(fake_Y)
        out_y = D_Y(fake_X)

        #same_X = G_YtoX(images_X)
        #same_Y = G_XtoY(images_Y)

        #id_X_loss = criterion_identity(same_X, images_X) * 5
        #id_Y_loss = criterion_identity(same_Y, images_Y) * 5

        g_YtoX_loss = mse_loss(out_x, real.expand_as(out_x))
        reconstructed_Y_loss = cycle_consistency_loss(reconstructed_Y, images_Y) * 10
        g_XtoY_loss = mse_loss(out_y, real.expand_as(out_y))
        reconstructed_X_loss = cycle_consistency_loss(reconstructed_X, images_X) * 10
        """geo_loss_X = silnet_loss(sil_fake_X, silhouette_X)
        geo_loss_Y = silnet_loss(sil_fake_Y, silhouette_Y)"""

        g_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_X_loss + reconstructed_Y_loss #+ geo_loss_X + geo_loss_Y #+ id_X_loss + id_Y_loss

        g_loss.backward()
        g_optimizer.step()

        for param in D_X.parameters():
            param.requires_grad = True
        for param in D_Y.parameters():
            param.requires_grad = True

        d_optimizer.zero_grad()

        pooled_fake_Y = G_XtoY.fake_pool.query(fake_Y)

        out_x_real = D_X(images_Y)
        d_loss_X_real = mse_loss(out_x_real, real.expand_as(out_x_real))

        out_x_fake = D_X(pooled_fake_Y.detach()) #D_X(fake_Y.detach())
        d_loss_X_fake = mse_loss(out_x_fake, fake.expand_as(out_x_fake))

        d_X_loss = (d_loss_X_real + d_loss_X_fake) * 0.5

        d_X_loss.backward()


        pooled_fake_X = G_YtoX.fake_pool.query(fake_X)
        out_y_real = D_Y(images_X)
        d_loss_Y_real = mse_loss(out_y_real, real.expand_as(out_y_real))

        out_y_fake = D_Y(pooled_fake_X.detach()) #D_Y(fake_X.detach())
        d_loss_Y_fake = mse_loss(out_y_fake, fake.expand_as(out_y_fake))

        d_Y_loss = (d_loss_Y_real + d_loss_Y_fake) * 0.5

        d_Y_loss.backward()

        d_optimizer.step()

        # Print the log info
        if it % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_X_loss.item(), d_Y_loss.item(), g_loss.item()))
            tqdm.write('Iteration [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    it, n_iters, d_X_loss.item(), d_Y_loss.item(), g_loss.item()))


        sample_every = 100#n_epochs/10
        # Save the generated samples
        if it % sample_every == 0:
            with torch.no_grad():
                G_YtoX.eval() # set generators to eval mode for sample generation
                G_XtoY.eval()
                save_samples(it, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=batch_size)
            G_YtoX.train()
            G_XtoY.train()
            checkpoint(it, G_XtoY, G_YtoX, D_X, D_Y)

    checkpoint(n_iters+1, G_XtoY, G_YtoX, D_X, D_Y)
    return losses




# ============================================
#                     MAIN
# ============================================


image_size = 256
batch_size = 8
n_res_blocks = 9

# hyperparams for Adam optimizer
lr = 0.0002
beta1 = 0.5
beta2 = 0.999 # default value

n_iters = 20000

# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
dataloader_X, test_dataloader_X = get_data_loader(image_dir='../data/synth2real', image_type='synth', batch_size=batch_size)
dataloader_Y, test_dataloader_Y = get_data_loader(image_dir='../data/synth2real', image_type='real', batch_size=batch_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# call the function to get models
G_XtoY, G_YtoX, D_X, D_Y = create_model(n_res_blocks=n_res_blocks, device=device)
"""S = SilNet()
S.load_state_dict(torch.load("silnet.pth"))
S.to(device)"""
#S.eval()

# print all of the models
print_models(G_XtoY, G_YtoX, D_X, D_Y, S)

G_XtoY.apply(weights_init_normal)
G_YtoX.apply(weights_init_normal)
D_X.apply(weights_init_normal)
D_Y.apply(weights_init_normal)



g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters()) # Get generator parameters
d_params = list(D_X.parameters()) + list(D_Y.parameters())

g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_optimizer = optim.Adam(d_params, lr, [beta1, beta2])
#sil_optimizer = optim.Adam(S.parameters(), lr, [beta1, beta2])

# Create learning rate schedulers for generators and discriminators
#g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=LambdaLR(n_epochs, 1, 100).step)
#d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=LambdaLR(n_epochs, 1, 100).step)
#s_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=LambdaLR(n_epochs, 1, 100).step)

# Lossess (SilNet and cycle-consistency losses are imported)
criterion_identity = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
cycle_consistency_loss = torch.nn.L1Loss()
silnet_loss = torch.nn.BCELoss()

real = torch.tensor(1.0, requires_grad=False).to(device)
fake = torch.tensor(0.0, requires_grad=False).to(device)

losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_iters=n_iters)

fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig("loss.png")
