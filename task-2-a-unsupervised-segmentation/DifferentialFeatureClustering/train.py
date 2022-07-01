import cv2
import sys
import random
import numpy as np

import torch
import argparse
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from network import MyNet
from dataset import CropData, LoadImages

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
    parser.add_argument('--scribble', action='store_true', default=False, 
                        help='use scribbles')
    parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                        help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                        help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                        help='number of convolutional layers')
    parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                        help='visualization flag')
    parser.add_argument('--input', metavar='images folder',
                        help='input the path containing image folder', required=True)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                        help='step size for similarity loss', required=False)
    parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                        help='step size for continuity loss')
    parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                        help='step size for scribble loss')
    parser.add_argument('--weights_path', type=str, help='path to save model weights')
    args = parser.parse_args()
    return args


args = arg_parser()

crop_images_array = LoadImages(args.input).load_images_into_array()

crop_dataset = CropData(
    images=crop_images_array
    )  #resize=True, dsize=(512, 512)


crop_dataloader = DataLoader(crop_dataset, batch_size=args.batch_size, shuffle=True)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyNet(input_dim=3, nChannel=args.nChannel, nConv=args.nConv)
model = model.train()
model = model.to(device=device)

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(data_loader):
  for batch_idx in range(1, args.maxIter+1):
    for X in data_loader:
      X = X.to(device=device)
      optimizer.zero_grad()
      output = model( X )
      output = output.permute( 0, 2, 3, 1).contiguous().view( -1, args.nChannel )

      HPy_target = torch.zeros(X.shape[0], X.shape[2]-1, X.shape[3], args.nChannel).to(device=device)
      HPz_target = torch.zeros(X.shape[0], X.shape[2], X.shape[3]-1, args.nChannel).to(device=device)

      outputHP = output.reshape( (X.shape[0], X.shape[2], X.shape[3], args.nChannel) )
      HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
      HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]

      HPy = HPy.to(device); HPz = HPz.to(device)

      lhpy = loss_hpy(HPy, HPy_target)
      lhpz = loss_hpz(HPz, HPz_target)

      _, target = torch.max( output, 1 )

      im_target = target.cpu().numpy()
      nLabels = len(np.unique(im_target))
            
      loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
                
      loss.backward()
      optimizer.step()

    print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

  torch.save(model.state_dict(), args.weights_path)

if __name__ == "__main__":
    train(crop_dataloader)

#python train.py --nChannel 100 --maxIter 1000 --minLabels 8 --visualize 0 --weights_path "weights.pt" --input Data\Data-20220522T105527Z-002\Barley

#python train.py --nChannel 5 --maxIter 20  --minLabels 4 --visualize 0 --weights_path "weights.pt" --input Data\Data-20220522T105527Z-002\Barley