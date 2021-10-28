import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable

from model import RegionNet
from dataset import ImageDataset

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--task_name", type=str, default="simple_region", help="name of the training task")
parser.add_argument("--size", type=int, default=128, help="size of the dataset images")
parser.add_argument("--n_class", type=int, default=32, help="number of classes of regions")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between sampling of images from model"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("path", type=str, help="dir path of the dataset")
opt, _ = parser.parse_known_args()
print(opt)

base_dir = os.path.dirname(__file__) 

os.makedirs(os.path.join(base_dir, "images/%s" % opt.task_name), exist_ok=True)
os.makedirs(os.path.join(base_dir, "saved_models/%s" % opt.task_name), exist_ok=True)


# Initialize model
model = RegionNet(n_class=opt.n_class).cuda()


if opt.epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load(os.path.join(base_dir, "saved_models/{}/model_{}.pth".format(opt.task_name, opt.epoch))))


# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform = transforms.Compose([
    transforms.Resize([opt.size, opt.size]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

dataloader = DataLoader(
    ImageDataset(opt.path, transform),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset(opt.path, transform),
    batch_size=4,
    shuffle=True,
    num_workers=0,
)


def sample_images(epoch, it):
    """Saves a generated sample from the validation set"""
    model.train()
    real_img = next(iter(val_dataloader))
    real_img = real_img.float().cuda()
    fake_img, fake_hard, region = model(real_img)
    rd_colors = torch.rand(4, opt.n_class, 3, 1, 1).cuda() * 2 - 1
    y = region[:,:,None,...]
    vis_region = y.float() * rd_colors.float()
    vis_region = vis_region.sum(dim=1) # b x c x h x w
    img_sample = torch.cat((real_img, fake_img, fake_hard, vis_region), 0).clamp(-1,1)
    save_image(
        img_sample, 
        os.path.join(base_dir, "images/%s/%s-%s.png" % (opt.task_name, str(epoch+1).zfill(3), str(it).zfill(8))), 
        nrow=4, 
        normalize=True, 
        range=(-1,1)
    )
    model.eval()


def save_fn(epoch):
    torch.save(model.state_dict(), os.path.join(base_dir, "saved_models/{}/model_{}.pth".format(opt.task_name, epoch)))


# ----------
#  Training
# ----------
epoch = opt.epoch
def train():
    global epoch
    for epoch in range(opt.epoch, opt.n_epochs):
        pbar = tqdm(dataloader)
        for i, real_img in enumerate(pbar):
            model.train()
    
            # Model inputs
            real_img = real_img.float().cuda()

            # ------------------
            #  Train Model
            # ------------------
    
            optimizer.zero_grad()
    
            # GAN loss
            fake_img = model(real_img)[0]
            loss_pixel = F.l1_loss(fake_img, real_img)
            
            loss_pixel.backward()
            optimizer.step()
    
            # Print log
            pbar.set_description(
                "\r[Epoch %d/%d] [loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    loss_pixel.item()
                )
            )
    
            # If at sample interval save image
            if i % opt.sample_interval == 0:
                sample_images(epoch, i)
    
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            save_fn(epoch)
            save_fn('latest')
        epoch += 1

if __name__ == '__main__':
    train()
