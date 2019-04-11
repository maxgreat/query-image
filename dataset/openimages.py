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
        self.classDict = {line.split(',')[0]:line.split(',')[1].rstrip() for line in open(dictFile)}

    def get_class_vector(self, c):
        if c in self.classDict:
            return self.model.get_word_vector(self.classDict[c])
        else:
            print("Word ", c, "not in class dictionnary")
            return self.model.get_word_vector(c)

    def get_class_from_id(self, c):
        if c in self.classDict:
            return self.classDict[c]
        else:
            print("Word ", c, "not in class dictionnary")
            return None
    
    def get_sentence_vector(self, sentence):
        return self.model.get_sentence_vector(sentence)
        
    def get_word_vector(self, word):
        return self.model.get_word_vector(word)
        
        

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
        im = Image.open(image_path).convert('RGB')
        return im.crop( (im.width*self.crop[0], im.height*self.crop[2], im.width*self.crop[1], im.height*self.crop[3]) )
        
        

class OpenImagesBbox(data.Dataset):
    def __init__(self, image_dir, bbox_file, classes,transform, sset="train", embeddings="/data/m.portaz/wiki.en.bin"):
        self.transform = transform
        self.image_dir = image_dir
        self.bbox_file = bbox_file
        self.wordEncode = WordEncoder(classes, embeddings)
        
        reader = csv.reader(open(bbox_file), delimiter=',')
        
        self.images = []
        for i, line in enumerate(reader):
            if i > 0:
                self.images.append(ImageBox(line))

        
    def __getitem__(self, index):
        image_path = self.image_dir + self.images[index].ID + '.jpg'
        im_emb = self.transform(self.images[index].getImage(image_path))
        txt_emb =  self.wordEncode.get_class_vector( self.images[index].label )
        return im_emb, txt_emb
        
        
    def __len__(self):
        return len(self.images) 
        
        


class OpenImages(data.Dataset):
    def __init__(self, image_dir, image_file, classes,transform, sset="train", embeddings="/data/m.portaz/wiki.en.bin"):
        self.transform = transform
        self.image_dir = image_dir
        self.image_file = image_file
        self.wordEncode = WordEncoder(classes, embeddings)
        
        reader = csv.reader(open(image_file), delimiter=',')
        
        self.images = []
        for i, line in enumerate(reader):
            if i > 0 and line[3] == '1':
                self.images.append( (line[0], self.wordEncode.get_class_from_id(line[2]) ) )

        
    def __getitem__(self, index):
        image_path = self.image_dir + self.images[index][0] + '.jpg'
        #return Image.open(image_path), self.wordEncode.get_class_from_id(self.images[index][1])
        im_emb = self.transform(Image.open(image_path))
        txt_emb = self.wordEncode.get_word_vector( self.images[index][1] )
        return im_emb, txt_emb
        
        
    def __len__(self):
        return len(self.images) 
        


