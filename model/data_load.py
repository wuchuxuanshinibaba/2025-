from torch.utils import data
from PIL import Image
import os
import numpy as np
import torch

class Dataset_all(data.Dataset):
    def __init__(self,x_txt_input_ids,x_txt_attention_masks,x_img,x_kg1,x_kg2,x_kg_sim,y,transform,pathset):
        self.transform = transform
        self.pathset = pathset
        img_dir = r"C:\Users\Administrator\Desktop\qimodazuoye\双重不一致谣言检测网络\dual-inconsistency-rumor-detection-network\data\pheme\images"
        print("图片文件夹路径:", img_dir)
        print("x_img[:10]:", x_img[:10])
        print("images文件夹下前10个文件:", os.listdir(img_dir)[:10])
        valid_indices = []
        for idx, img_name in enumerate(x_img):
            name = str(img_name).strip()
            # 去掉扩展名
            if name.lower().endswith('.jpg'):
                name = name[:-4]
            elif name.lower().endswith('.png'):
                name = name[:-4]
            if any(os.path.isfile(os.path.join(img_dir, name + ext)) for ext in ['.jpg', '.png']):
                valid_indices.append(idx)
        self.data_txt_input_ids = [x_txt_input_ids[i] for i in valid_indices]
        self.data_txt_attention_masks = [x_txt_attention_masks[i] for i in valid_indices]
        self.data_img = [x_img[i] for i in valid_indices]
        self.data_kg1 = [x_kg1[i] for i in valid_indices]
        self.data_kg2 = [x_kg2[i] for i in valid_indices]
        self.data_kg_sim = [x_kg_sim[i] for i in valid_indices]
        self.label = [y[i] for i in valid_indices]

    def __getitem__(self,index):
        img_name = str(self.data_img[index]).strip()
        # 去掉扩展名
        if img_name.lower().endswith('.jpg'):
            img_name = img_name[:-4]
        elif img_name.lower().endswith('.png'):
            img_name = img_name[:-4]
        img_dir = r"C:\Users\Administrator\Desktop\qimodazuoye\双重不一致谣言检测网络\dual-inconsistency-rumor-detection-network\data\pheme\images"
        img_path_jpg = os.path.join(img_dir, img_name + '.jpg')
        img_path_png = os.path.join(img_dir, img_name + '.png')
        if os.path.isfile(img_path_jpg):
            image = Image.open(img_path_jpg).convert('RGB')
        elif os.path.isfile(img_path_png):
            image = Image.open(img_path_png).convert('RGB')
        else:
            raise FileNotFoundError(f"Image file not found: {img_path_jpg} or {img_path_png}")
        image = self.transform(image)
        return self.data_txt_input_ids[index],self.data_txt_attention_masks[index], image,self.data_kg1[index],self.data_kg2[index],self.data_kg_sim[index],self.label[index]
    def __len__(self):
        return len(self.data_txt_input_ids)