import os
import glob


# 使用FreeSurfer对图像进行颅骨剥离
print("FreeSurfer start......\n")
# 图像坐在的目录
#------------------------图像路径需更改------------------------#
path ="/home/user/Documents/dataset/oasis/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111"
# 读取目录下的.img文件列表，*.img表示该目录下所有以.img结尾的文件
#-----------------------图像后缀名需更改--- -------------------#
files = glob.glob(os.path.join(path,"*.img"))
# 输出路径
#------------------------输出路径需更改------------------------#
out_path="/home/user/Documents/dataset/oasis/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/process"
print("number: ",len(files))
# 下面为freesurfer的环境配置命令
a = "export FREESURFER_HOME=/usr/local/freesurfer/7.3.2;"
b = "source $FREESURFER_HOME/SetUpFreeSurfer.sh;"
# 数据所在的目录
c = "export SUBJECTS_DIR="+path+";"

for file in files:
    # 将文件路径和文件名分离
    filename = os.path.split(file)[1] # 将路径名和文件名分开
    filename = filename.split(".")[0] # 去除所有扩展名   
    #recon-all是颅骨去除的命令
    # mri_convert是进行格式转换，从mgz转到nii.gz，只是为了方便查看
    filename=filename[:] #根据扩展名的不同，这里需要做更改，只保留文件名即可
    # 当前输出文件路径，以.nii.gz格式保存
    cur_out_path=os.path.join(out_path,filename+"_rmsk.nii.gz")
    print("file name: ",file)
    cmd = a + b + c + "mri_watershed "+file+" "+ cur_out_path
    #print(cmd,"\n")
    os.system(cmd)
    cur_out_path1=os.path.join(out_path,filename+"_ori.nii.gz")
    cmd = a + b + c + "mri_convert " +file+" "+ cur_out_path1
    #print(cmd,"\n")
    os.system(cmd)


# 使用FSL对图像和对应的label进行仿射对齐
print("FSL start......\n")
# fixed图像的路径
#---------------去除头骨后的fixed图像名需更改-------------------#
f_path= "/home/user/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
# moving图像的路径
m_path=out_path
# label的路径
#-----------------------label路径需更改-----------------------#
# label_path="/home/syzhou/zuzhiang/Dataset/MGH10/Atlases"
files=glob.glob(os.path.join(m_path,"*.nii.gz"))
print("number: ",len(files))
for file in files:
    print("file: ",file)
    # 根据图像名找到对应的label名
    filename=os.path.split(file)[1]
    filename = filename.split(".")[0] # 去除所有扩展名
    #---------------------label后缀名需更改--------------------#
    # label=os.path.join(label_path,filename+".img")
    # 下面分别是输出图像名/转换矩阵名/label名，
    out_img=os.path.join(out_path,filename+"_aligned.nii.gz")
    out_mat=os.path.join(out_path,filename+".mat")
    # out_label=os.path.join(out_path,filename+"_label.nii.gz")
    # 如果当前文件和fixed图像一样则只将对应label格式转换为.nii.gz 
    # 将moving图像对齐到fixed图像
    flirt_img="flirt -in "+file+ " -ref "+f_path+" -out "+out_img+" -omat "+out_mat+ " -dof 6"
    # 将上一步的仿射变换矩阵作用在图像对应的label上
    # flirt_label="flirt -in "+label+" -ref "+f_path+" -out "+out_label+" -init "+out_mat+" -applyxfm -interp nearestneighbour"
    #print(flirt_img,"\n")
    #print(flirt_label,"\n")
    os.system(flirt_img)
    # os.system(flirt_label)

print("\n\nEnd")
