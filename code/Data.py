import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os

class Data_Read_Preprocessing(nn.Module):
    def __init__(self, folder_path, istransform = True, **kwargs):
        super(Data_Read_Preprocessing, self).__init__(**kwargs)
        self.istransform = istransform
        self.folder_path = folder_path
        self.transform = Compose([Resize((224, 224)), ToTensor()])

    def readImages(self):
        return [os.path.join(self.folder_path, filename)
                for filename in os.listdir(self.folder_path)
                if filename.lower().endswith(('jpg', 'jpeg'))]

    def load_and_process(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            if self.istransform:
                image = self.transform(image)
            else:
                image = ToTensor()(image)
            return image.unsqueeze(0)
        except IOError:
            raise ValueError(f"Could not open or read the image file {image_path}.")

    def forward(self):
        image_paths = self.readImages()
        transformed_images = []  # 创建一个空列表来存储所有处理过的图像
        for image_path in image_paths:
            image = self.load_and_process(image_path)
            transformed_images.append(image)  # 将每个处理过的图像添加到列表中
        return transformed_images

# data = Data_Read_Preprocessing('E:\\pytorch_learning\\vit_pytorch\\data')
# print(data.forward()[0].shape) # 返回 [1,3,224,224]
