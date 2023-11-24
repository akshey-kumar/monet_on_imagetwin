import numpy as np
import torch
import pickle
import json
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset

class MonetDL(Dataset):

    def __init__(self,
                 dataset = 'train'):

        self.dataset = dataset
        self.image_dir = './dataset/data_' + self.dataset
        self.sample_list = self.load_samples()
        
        with open(os.path.join(self.image_dir, 'labels_' + self.dataset), 'r') as f:
            self.labels = np.loadtxt(f).astype(int)


    def load_samples(self):

        sample_list = []
        directory_path = os.path.join(self.image_dir, 'features_' + self.dataset)
        feature_directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        
        for i in range(len(feature_directories)):
            
            folder_name = os.path.join(directory_path, str(i))

            panel1 = os.path.join(folder_name, 'img_1.png')
            panel2 = os.path.join(folder_name, 'img_2.png')
            
            sample_list += [(str(i), panel1, panel2)]
                        
        return sample_list

    def preprocess_img(self, img):

        prep_img = cv2.resize(img, (256,256))
        prep_img = prep_img/255.
        prep_img = (prep_img-0.5)/0.5
        prep_img = torch.tensor(prep_img).permute(2,0,1)

        return prep_img

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, idx):

        doi, panel1, panel2 = self.sample_list[idx]
        img1 = cv2.imread(panel1)
        img2 = cv2.imread(panel2)
        
        ### img1 img2 are our image pair from the dataset

        s1 = torch.tensor(img1.shape)
        s2 = torch.tensor(img2.shape)
        
        img1 = self.preprocess_img(img1)
        img2 = self.preprocess_img(img2)

        return img1.float(), img2.float(), s1, s2, doi, panel1, panel2


### Testing
if __name__ == '__main__':

    train_dl = MonetDL(dataset = 'train')
    print(type(train_dl))
    print(train_dl)
    print(train_dl.__getitem__(5))


    


