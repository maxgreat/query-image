import fastText
import torch
import torch.utils.data as data
import csv
from PIL import Image
from nltk.tokenize import word_tokenize

idx_ID = 0
idx_Label = 2
idx_X0 = 4
idx_X1 = 5
idx_Y0 = 6
idx_Y1 = 7


def load_vec(emb_path):
    """
        Load FastText model .vec
        Returns : embeddings (matrix index -> embedding), id2word ( index -> word ) and word2id ( word -> index)
    """
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id



class WordEncoder:
    def __init__(self, dictFile, vecFile):
        self.model = fastText.load_model(vecFile)

    def get_sentence_vector(self, sentence):
        return self.model.get_sentence_vector(sentence)
        

class ImageBox:
    def __init__(self, line):
        self.ID = line[idx_ID]
        self.label = line[idx_Label]
        self.crop = ( float(line[idx_X0]), 
                    float(line[idx_X1]),
                    float(line[idx_Y0]), 
                    float(line[idx_Y1])
                    )

    def getImage(self, image_path):
        print(self.ID)
        im = Image.open(image_path)
        return im.crop( (im.width*self.crop[0], im.height*self.crop[2], im.width*self.crop[1], im.height*self.crop[3]) ), self.label
        
        

class OpenImages(data.Dataset):
    def __init__(self, image_dir, bbox_file, classes,transform, sset="train", embeddings="/data/m.portaz/wiki.en.bin"):
        self.transform = transform
        self.image_dir = image_dir
        self.bbox_file = bbox_file
        self.wordEncode = WordEncoder(classes, embeddings)
        
        reader = csv.reader(open(bbox_file), delimiter=',')
        
        self.images = []
        for i, line in enumerate(reader):
            if i > 0 and i == 1:
                self.images.append(ImageBox(line))

        
    def __getitem__(self, index):
        image_path = self.image_dir + self.images[index].ID + '.jpg'
        
        return self.images[index].getImage(image_path), self.wordEncode.get_sentence_vector(self.images[index].label)
        
        
    def __len__(self):
        return len(self.images) 
