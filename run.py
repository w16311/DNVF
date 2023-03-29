from train import *
from Utils import *
import nibabel as nib

def main():
    fixed = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/image_in_MNI152_normalized/OASIS-TRT-20-1.nii.gz')
    moving = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/image_in_MNI152_normalized/OASIS-TRT-20-12.nii.gz')
    # fixed = load_nii('/home/user/Documents/dataset/oasis_seg/OASIS_OAS1_0005_MR1/aligned_norm.nii.gz')
    # moving = load_nii('/home/user/Documents/dataset/oasis_seg/OASIS_OAS1_0011_MR1/aligned_norm.nii.gz')
    # fixed = fixed[9:173, 13:205, 19:163]
    # moving = moving[9:173, 13:205, 19:163]
    fixed_seg = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged/OASIS-TRT-20-1.nii.gz')
    moving_seg = load_nii('/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged/OASIS-TRT-20-12.nii.gz')
    # fixed_seg = load_nii('/home/user/Documents/dataset/oasis_seg/OASIS_OAS1_0005_MR1/aligned_seg35.nii.gz')
    # moving_seg = load_nii('/home/user/Documents/dataset/oasis_seg/OASIS_OAS1_0011_MR1/aligned_seg35.nii.gz')
    # fixed_seg = fixed_seg[9:173, 13:205, 19:163]
    # moving_seg = moving_seg[9:173, 13:205, 19:163]
    
    im_warped, seg_warped, dice = train([fixed, moving], [fixed_seg, moving_seg], T = 7, n_epoch=300)
    tmp = nib.load('/home/user/Documents/dataset/Mindboggle101/mindboggle/label_31_reID_merged/OASIS-TRT-20-1.nii.gz')
    seg_affine = tmp.affine
    nib.Nifti1Image(im_warped,seg_affine).to_filename('im_warped.nii.gz')
    nib.Nifti1Image(seg_warped,seg_affine).to_filename('seg_warped.nii.gz')


    print(dice)

if __name__ == '__main__':
    main()