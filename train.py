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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train(img, seg, label, T = 7, n_epoch=300):
    setup_seed(1357)
    device = torch.device('cuda:0')
    mlp = Siren(in_features=3, out_features=3, hidden_features=512, 
                      hidden_layers=3, outermost_linear=True)
    mlp.to(device)
    optimizer = torch.optim.Adam(lr=1e-4, params=mlp.parameters(), amsgrad=True)
    loss_NCC = NCC(win=15)

    # load img
    fixed = img[0]
    moving = img[1]
    moving = torch.from_numpy(moving).to(device).float()
    fixed = torch.from_numpy(fixed).to(device).float()
    # make batch dimension
    im_m = moving.unsqueeze(0).unsqueeze(0)
    im_f = fixed.unsqueeze(0).unsqueeze(0)

    # load label
    seg_f = seg[0]
    moving_seg = seg[1]
    moving_seg = torch.from_numpy(moving_seg).to(device).float()
    # make batch dimension
    seg_m = moving_seg[None, None, ...]

    # generate grid
    im_shape = list(im_f.squeeze().shape)
    grid_f = generate_grid3D_tensor(im_shape, factor=1, sq=False).unsqueeze(0).to(device)  # [-1,1]
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1]
    mgrid_shape = []
    mgrid_shape.append(3)
    for i in im_shape:
        mgrid_shape.append(int(i/2))
    # label = list(range(32))

    # original dice loss
    dice_move2fix = dice(seg_m.unsqueeze(0).detach().cpu().numpy(), seg_f, label)
    print("{:=^50s}".format("Start epoch"))
    print('Original Avg. dice on %d structures: ' % len(label), np.mean(dice_move2fix))
    loss_mse = torch.nn.MSELoss(reduction='mean')

    # train epoch
    for i in range(n_epoch):
        df, _ = mlp(grid) # deformation field N*3 N is number of point
        
        # grid field 3*H*W*D
        
        phi = 0.1 * df + grid
        phi = phi.squeeze().T.reshape(mgrid_shape)
        df = df.squeeze().T.reshape(mgrid_shape)
        
        df_up =  F.grid_sample(df.unsqueeze(0), grid_f, mode='bilinear',align_corners=True) #Up sample, interpolate

        # Diffeomorphic Deformation as Integration (4.2 section of paper)
        df_cur = grid_f + 0.1 * df_up.permute(0, 2, 3, 4, 1) / torch.pow(torch.tensor(2), T) 
        df_cur = F.grid_sample(df.unsqueeze(0), df_cur, mode='bilinear',align_corners=True)        
        for j in range(T, 0, -1):
             df_next = grid_f + 0.1 * df_cur.permute(0, 2, 3, 4, 1) / torch.pow(torch.tensor(2), j)
             df_cur = F.grid_sample(df.unsqueeze(0), df_next, mode='bilinear',align_corners=True)
        grid_warped = grid_f + 0.1 * df_cur.permute(0, 2, 3, 4, 1)


        # No Integration
        # grid_warped = grid_f + 0.1 * df_up.permute(0, 2, 3, 4, 1)


        im_warped = F.grid_sample(im_m, grid_warped, mode='bilinear',align_corners=True)
        # import pdb;pdb.set_trace()
        # similarity loss
        loss_sim = loss_NCC(im_warped, im_f)
        # loss_sim = loss_mse(im_warped, im_f)
        # neg Jacobian loss
        # loss_J = 100 * neg_Jdet_loss(df.unsqueeze(0).permute(0, 2, 3, 4, 1))
        loss_J = 100 * neg_Jdet_loss(phi.unsqueeze(0).permute(0, 2, 3, 4, 1))
        # phi dphi/dx loss
        loss_df = 0.1 * smoothloss_loss(df.unsqueeze(0))
        loss = loss_sim + loss_J + loss_df
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # import pdb;pdb.set_trace()
        

        if (i + 1) % 20 == 0 or i==0:
            neg_jet = 100*neg_Jdet(phi.unsqueeze(0).permute(0, 2, 3, 4, 1))
            print("Iteration: {0} Loss_sim: {1:.3e} loss_J: {2:.3e} ".format(i + 1, loss_sim.item(), loss_J.item()))
            print('Ratio of neg Jet: ', neg_jet.detach().cpu().numpy())
            seg_warped = F.grid_sample(seg_m, grid_warped, mode='nearest', align_corners=True)
            # import pdb;pdb.set_trace()
            dice_move2fix = dice(seg_warped.unsqueeze(0).unsqueeze(0).detach().cpu().numpy(), seg_f, label)
            print('Avg. dice on %d structures: ' % len(label), np.mean(dice_move2fix))
    im_warped = im_warped.squeeze().detach().cpu().numpy()
    seg_warped = seg_warped.squeeze().detach().cpu().numpy()
    loss_all = [loss_sim.detach().cpu().numpy(), loss_J.detach().cpu().numpy(),\
                loss_df.detach().cpu().numpy()]
    return  im_warped, seg_warped, dice_move2fix, loss_all