import argparse
import sys
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt


import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./test_images', help='root directory of the test dataset')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(3, 3)
netG_B2A = Generator(3, 3)


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load('weights/netG_A2B.pth', map_location=torch.device('cpu')))
netG_B2A.load_state_dict(torch.load('weights/netG_B2A.pth', map_location=torch.device('cpu')))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(1, 3, opt.size, opt.size)
input_B = Tensor(1, 3, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

transforms_ = transforms.Compose(transforms_)                
###################################

###### Testing######

As = glob.glob(os.path.join(opt.dataroot, '*.jpg'))

plt.figure(figsize=(12, 6))
fixed_size = (200, 200)

for i in range(len(As)):
    img = Image.open(As[i])
    real_A = transforms_(img)
    size_A = real_A.shape
    real_A = transforms.Resize([opt.size,opt.size])(real_A)
    real_A = real_A.unsqueeze(0)
    real_A = Variable(input_A.copy_(real_A))
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_B = fake_B.squeeze()
    
    plt.subplot(2, 4, 1 + i * 2)
    # plt.subplot(2, 4, 1 + i)
    plt.imshow(img.resize(fixed_size), cmap="binary")
    plt.axis("off")
    plt.subplot(2, 4, 2 + i * 2)
    # plt.subplot(2, 4, 1 + len(As) + i)
    fake_B = transforms.ToPILImage()(fake_B).resize(fixed_size)
    plt.imshow(fake_B, cmap="binary")
    plt.axis("off")
    print(f'\rGenerated Bs, {i+1} of {len(As)}', end='')

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.savefig('results.png')

