from glob import glob
from torch.utils.data import Dataset
import os
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
            T.PILToTensor(),
            T.Resize([320,320])
            ])


class ExteriorDataset(Dataset):
    def __init__(self,root):
        self.root = root
        self.imgs = glob(os.path.join(self.root,"images","*.png"))+glob(os.path.join(self.root,"images","*.jpg"))\
                    +glob(os.path.join(self.root,"images","*.jpeg"))+glob(os.path.join(self.root,"images","*.PNG"))\
                    +glob(os.path.join(self.root,"images","*.JPG"))+glob(os.path.join(self.root,"images","*.JPEG"))
        self.masks = glob(os.path.join(self.root,"masks","*.png"))+glob(os.path.join(self.root,"masks","*.jpg"))\
                    +glob(os.path.join(self.root,"masks","*.jpeg"))+glob(os.path.join(self.root,"masks","*.PNG"))\
                    +glob(os.path.join(self.root,"masks","*.JPG"))+glob(os.path.join(self.root,"masks","*.JPEG"))
        noimgs = len(self.imgs)
        nomasks = len(self.masks)
        print(f"\tnum of images given = {noimgs} , num of masks given = {nomasks}")
        self.new_imgs, self.new_masks = [], []
        self.imgs.sort()
        self.masks.sort()
        if nomasks<=noimgs:
            i=0
            j=0
            while i<nomasks and j<noimgs:
                if self.masks[i].replace("masks","images").split(".")[:-1] == self.imgs[j].split(".")[:-1]:
                    self.new_masks.append(self.masks[i])
                    self.new_imgs.append(self.imgs[j])
                    i+=1
                    j+=1
                elif self.masks[i].replace("masks","images").split(".")[:-1] < self.imgs[j].split(".")[:-1]:
                    i+=1
                else:
                    j+=1
        else:
            i=0
            j=0
            while i<noimgs and j<nomasks:
                if self.imgs[i].replace("images","masks").split(".")[:-1] == self.masks[j].split(".")[:-1]:
                    self.new_imgs.append(self.imgs[i])
                    self.new_masks.append(self.masks[j])
                    i+=1
                    j+=1
                elif self.imgs[i].replace("images","masks").split(".")[:-1] < self.masks[j].split(".")[:-1]:
                    i+=1
                else:
                    j+=1
        print(f"\tnum of images= {len(self.new_imgs)} , num of masks = {len(self.new_masks)}")
    def __len__(self):
        return len(self.new_imgs)
    def __getitem__(self, index):
        impath = self.new_imgs[index]
        maskpath = self.new_masks[index]
        im = Image.open(impath)
        mask = Image.open(maskpath)
        im = transform(im)/255
        mask = transform(mask)/255
        # print(im.shape,mask.shape,maskpath)
        return im,mask




# path = "/home/mukesh/Desktop/Spyne/ai/training/U2_net/training_data"
# ds = ExteriorDataset(path)
# print(ds.__getitem__(1))
