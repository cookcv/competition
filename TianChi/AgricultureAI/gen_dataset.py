import os
from tqdm import tqdm
import openslide
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
# from itertools import groupby
# from skimage import measure
from matplotlib import image

                        
# %matplotlib inline

def get_label(mask):
    """通过传入的mask判断所属类别"""
    """可能同时存在三种类别，所以定义返回值为list"""

    return [label for label in [1,2,3] if mask[mask==label].size > 0 ]


def convert(slide_image,slide_mask,img_dir,mask_dir,root_img_name,stride=512,img_size=600):

    """
    :param slide_image  :the big root_image 
    :param slide_mask   :the big root_mask 
    :param img_dir      :crop_train_img_dataset_dir
    :param mask_dir     :crop_train_mask_dataset_dir
    :param root_img_name:原卫星图片的名字
    :param stride       :滑动步长
    :param img_size     :剪切图片大小
    """

    width,height = slide_image.dimensions
    root_img_name = root_img_name.split('.')[0] # 去除.png后缀
    
    for w_s in tqdm(range(0,width,stride)):
        for h_s in range(0,height,stride):
            
            img = np.array(slide_image.read_region((w_s,h_s), 0, (img_size,img_size) ))[:,:,:3]
            if len(np.flatnonzero(img))/(img_size*img_size) < 0.75: # 图像占比0.75才保存图像
                continue
            
            image_name = root_img_name + "_" +str(w_s) + "_" + str(h_s) 

            if slide_mask:
                mask_label = np.array(slide_mask.read_region((w_s,h_s), 0, (img_size,img_size) ))
            
                label_list = get_label(mask_label[:,:,:1])
                if not label_list: # 如果该mask中没有标签，则跳过
                    continue
                # mask_label = np.squeeze(mask_label)# 降维
                image.imsave(os.path.join(mask_dir, image_name+".png"), mask_label)

            # 保存切片图像
            image.imsave(os.path.join(img_dir, image_name+".jpg"), img)
            
            
def creat_dataset(root_dir,root_img_name,root_mask_name,img_dir_save,mask_dir_save):
    
    """
    :param root_dir      :图片数据与mask数据的根目录
    :param root_img_name :原卫星图片的名字
    :param root_mask_name:原卫星mask图片的名字,测试集没有为False
    :param img_dir_save  :切片图片保存的路径
    :param mask_dir_save :切片mask图片保存的路径,测试集没有为False
    """
    slide_image = openslide.open_slide(os.path.join(root_dir, root_img_name)) # image_1.png
    slide_mask = False
    if root_mask_name:
        slide_mask = openslide.open_slide(os.path.join(root_dir, root_mask_name)) # image_1_label.png
        
    convert(slide_image,slide_mask,img_dir_save,mask_dir_save,root_img_name)
        
if __name__ == "__main__":
    
    Image.MAX_IMAGE_PIXELS = 3000000000

    train_dir = '../data/jingwei_round1_train_20190619/'
    img_train_dir_save = "../data/unet_train/image/"
    mask_train_dir_save = "../data/unet_train/label/"
    
    test_dir = '../data/jingwei_round1_test_a_20190619/'
    img_test_dir_save = "../data/unet_test/image/"
    
    # 创建保存目录
    for file_dir in [img_train_dir_save,mask_train_dir_save,img_test_dir_save]:
        if not os.path.exists(file_dir): os.makedirs(file_dir)
            
    train_imge_list = os.listdir(train_dir)
    test_imge_list = os.listdir(test_dir)
    for img_index in range(0,len(train_imge_list),2):
        
        '''train_dataset'''
        img_name = train_imge_list[img_index]
        mask_name = train_imge_list[img_index+1]
        
        creat_dataset(train_dir,img_name,mask_name,img_train_dir_save,mask_train_dir_save)

        # '''test_dataset'''
        # img_name = test_imge_list[img_index]
        # mask_name= False
        # mask_test_dir_save=False
        # creat_dataset(test_dir,img_name,mask_name,img_test_dir_save,mask_test_dir_save)
        
    
    