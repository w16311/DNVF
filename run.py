import os
import glob
from train import *
from Utils import *
import nibabel as nib
import numpy as np

def main():
    path_img = "/home/user/Documents/dataset/Mindboggle101/mindboggle/image_in_MNI152_normalized"
    path_seg = "/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged"
    out_path = "T7_Jdet100/"
    out_path = os.path.join(os.getcwd(),out_path)
    if not os.path.exists(out_path):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_path)
    img_file = []
    seg_file = [] 
    dice_all = []
    loss_all = []
    files = glob.glob(os.path.join(path_seg,"*.nii.gz"))
    for file in files:
        if "flipped" not in file:
            seg_file.append(file)
            tmp = os.path.split(file)
            tmp = os.path.join(path_img,tmp[1])
            img_file.append(tmp)
    # label = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    label = list(range(32))
    fixed = load_nii(img_file[0])
    fixed_seg = load_nii(seg_file[0])
    num_train = 61
    for i in range(1, num_train):
        moving = load_nii(img_file[i])
        moving_seg = load_nii(seg_file[i])
        im_warped, seg_warped, dice, loss = train([fixed, moving], [fixed_seg, moving_seg], label, T = 7, n_epoch=300)
        dice_all.append(dice)
        loss_all.append(loss)
        # save result
        tmp = nib.load(img_file[i])
        seg_affine = tmp.affine
        filename = os.path.split(img_file[i])[1]
        filename = filename.split(".")[0]
        nib.Nifti1Image(im_warped,seg_affine).to_filename(out_path + filename + '_im_warped.nii.gz')
        nib.Nifti1Image(seg_warped,seg_affine).to_filename(out_path + filename + 'seg_warped.nii.gz')
        # break
    np.savez(out_path + "result",dice = np.array(dice_all),loss = np.array(loss_all))


    # save result
    # tmp = nib.load('/home/user/Documents/NODEO-DIR/data/OAS1_0001_MR1/brain_aseg.nii.gz')
    # seg_affine = tmp.affine
    # nib.Nifti1Image(im_warped,seg_affine).to_filename('im_warped.nii.gz')
    # nib.Nifti1Image(seg_warped,seg_affine).to_filename('seg_warped.nii.gz')



if __name__ == '__main__':
    main()