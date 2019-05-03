
# coding: utf-8

# In[3]:


from nltk.corpus import wordnet
import os.path as pth
import glob


# In[43]:


import fastText
import torch
import torch.utils.data as data
import csv
from PIL import Image
from nltk.tokenize import word_tokenize
import random
import torchvision.transforms as transforms


# In[60]:


class FullImageNet(data.Dataset):
    def __init__(self, main_dir, transform, word_enc = fastText.load_model("/data/m.portaz/wiki.en.bin"), 
                 sset="train"):
        self.imageList = []
        self.sset = sset
        self.ts = transform
        self.word_enc = word_enc
        for directory in glob.iglob(pth.join(main_dir + '*')):
            if pth.isdir(directory):
                d = pth.basename(directory)
                s = wordnet.synset_from_pos_and_offset(d[0], int(d[1:]))
                
                for image in glob.iglob(pth.join(directory, "*.jpg")):
                    if sset == "train":
                        self.imageList.append( (image, s.lemma_names()) )
                    else :
                        self.imageList.append( (image, [s.lemma_names()[0]]) )
                    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        im, cl = self.imageList[index]
        txt_emb = self.word_enc.get_sentence_vector( random.choice(cl).replace("_", " ") )
        return self.ts(Image.open(im).convert("RGB")), txt_emb


# In[61]:


if __name__ == "__main__":
    main_dir = "/data/datasets/imageNet/images/"
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prepro = transforms.Compose([
        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    d = FullImageNet(main_dir, transform=prepro)
    print(d[0][0].shape, d[0][1].shape)

