import fastText
import torch
import torch.utils.data as data
import csv
from PIL import Image

idx_ID = 0
idx_Label = 2
idx_X0 = 5
idx_X1 = 6
idx_Y0 = 7
idx_Y1 = 7


class ImageBox:
    def ___int__(self, line):
        self.ID = line[idx_ID]
        self.label = line[idx_label]
        self.X0 = line[idx_X0]
        self.X1 = line[idx_X1]
        self.Y0 = line[idx_Y0]
        self.Y1 = line[idx_Y1]

    def getImage(image_path):
        Image.open(image_path)
        Image.
        

class OpenImages(data.Dataset):
    def __init__(self, image_dir, bbo_file, classes, transform, sset="train"):
        self.transform = transform
        self.image_dir = image_dir
        self.bbox_file = bbox_file
        self.classes = classes
        
        reader = csv.reader(open(bbox_file))
        
        for i, line in enumerate(reader):
            if i > 0:
                self.bbox = ImageBox

        

    def __getitem__(self, index):
        
    def __len__(self):
        return len(self.imList)
