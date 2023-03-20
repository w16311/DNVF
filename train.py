from dataset import *
import os
import time
import datetime
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from Utils import *
from Loss import *

from model import Siren

def main():
    crop_size=[0, 10, 7, 14, 8, 7]
    device = torch.device('cuda:0')
    mlp = Siren(in_features=3, out_features=3, hidden_features=512, 
                      hidden_layers=3, outermost_linear=True)
    mlp.to(device)
    optimizer = torch.optim.Adam(lr=1e-4, params=mlp.parameters(), amsgrad=True)
    loss_NCC = NCC(win=21)
    train_transforms = []
    train_transforms.append(SitkToTensor())
    if crop_size:
            train_transforms.append(CropTensor(crop_size))
    train_transforms = transforms.Compose(train_transforms)
    training_data = RegDataSetMindBoggle('/home/user/Documents/dataset/Mindboggle101/mindboggle/test.txt', 
                                '/home/user/Documents/dataset/Mindboggle101/mindboggle', with_seg=True,
                                preload=False, pre_transform=train_transforms,
                                n_samples=21 * 2)
    training_data_loader = DataLoader(training_data, batch_size=1,
                                    shuffle=True, num_workers=0)
    train_data_iter = iter(training_data_loader)
    # moved, fixed = next(train_data_iter)# moved[0] MRI images moved[1] seg label
    fixed = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/image_in_MNI152_normalized/OASIS-TRT-20-12.nii.gz')
    moving = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/image_in_MNI152_normalized/OASIS-TRT-20-2.nii.gz')
    moving = torch.from_numpy(moving).to(device).float()
    fixed = torch.from_numpy(fixed).to(device).float()
    # make batch dimension
    im_m = moving.unsqueeze(0).unsqueeze(0)
    im_f = fixed.unsqueeze(0).unsqueeze(0)
    fixed_seg = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged/OASIS-TRT-20-12.nii.gz')
    moving_seg = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged/OASIS-TRT-20-2.nii.gz')
    seg_f = fixed_seg
    moving_seg = torch.from_numpy(moving_seg).to(device).float()
    # make batch dimension
    seg_m = moving_seg[None, None, ...]
    im_shape = list(im_f.squeeze().shape)
    # import pdb;pdb.set_trace()
    grid_f = generate_grid3D_tensor(im_shape, factor=1, sq=False).unsqueeze(0).to(device)  # [-1,1]
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1]
    # import pdb;pdb.set_trace()
    mgrid_shape = []
    mgrid_shape.append(3)
    for i in im_shape:
        mgrid_shape.append(int(i/2))
    label = list(range(32))
    dice_move2fix = dice(seg_m.unsqueeze(0).detach().cpu().numpy(), seg_f, label)
    print('Original Avg. dice on %d structures: ' % len(label), np.mean(dice_move2fix))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for i in range(100):
        df, _ = mlp(grid) # deformation field N*3 N is number of point
        
        # grid field 3*H*W*D
        # import pdb;pdb.set_trace()
        df = df.squeeze().T.reshape(mgrid_shape)
        df_up =  F.grid_sample(df.unsqueeze(0), grid_f, mode='bilinear',align_corners=True) #Up sample, interpolate
        grid_warped = grid_f + 0.1 * df_up.permute(0, 2, 3, 4, 1)
        im_warped = F.grid_sample(im_m, grid_warped, mode='bilinear',align_corners=True)
        # import pdb;pdb.set_trace()
        # similarity loss
        loss_sim = loss_NCC(im_warped, im_f)
        # loss_sim = loss_fn(im_warped, im_f)
        # neg Jacobian loss
        loss_J = 100 * neg_Jdet_loss(df.unsqueeze(0).permute(0, 2, 3, 4, 1))
        # phi dphi/dx loss
        loss_df = 0.1 * smoothloss_loss(df.unsqueeze(0))
        loss = loss_sim + loss_J + loss_df
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # import pdb;pdb.set_trace()
        

        if (i + 1) % 20 == 0 or i==0:
            neg_jet = neg_Jdet(df.unsqueeze(0))
            print("Iteration: {0} Loss_sim: {1:.3e} loss_J: {2:.3e} ".format(i + 1, loss_sim.item(), loss_J.item()))
            print('Ratio of neg Jet: ', neg_jet)
            seg_warped = F.grid_sample(seg_m, grid_warped, mode='nearest', align_corners=True)
            # seg_warped = F.grid_sample(seg_m, grid_warped, mode='bilinear', align_corners=True)
            # seg_warped = torch.round(seg_warped)
            # import pdb;pdb.set_trace()
            dice_move2fix = dice(seg_warped.unsqueeze(0).unsqueeze(0).detach().cpu().numpy(), seg_f, label)
            print('Avg. dice on %d structures: ' % len(label), np.mean(dice_move2fix))
            # break

        

if __name__ == '__main__':
    main()