import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
from torchvision import  transforms

def create_collate_fn(word2idx):
    def collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, img, caption, caplen in dataset:           
            tensor_caption=caption.unsqueeze(0)
            caption = caption.numpy().tolist()
            caption = [w for w in caption if w != word2idx['<pad>']]
            ground_truth[fn] = [caption[:]]
            for cap in [caption]:
                tmp.append([fn, img.unsqueeze(0),tensor_caption, cap, caplen])       
        dataset = tmp  
        dataset.sort(key=lambda p: len(p[3]),
                     reverse=True)  
        fns, imgs, tensor_captions,caps, caplens = zip(*dataset)
        imgs = torch.cat((imgs), dim=0)

        lengths = [min(len(c), 52) for c in caps]
        caps_tensor = torch.cat((tensor_captions),dim=0)  
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])
        lengths = torch.LongTensor(lengths)
        lengths = lengths.unsqueeze(1)
        return fns, imgs, caps_tensor, lengths, ground_truth  
    return collate_fn

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_name, split, loadpath,
                 transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.transform_RandomRotation = transforms.Compose([transforms.RandomRotation(360)])
        self.transform_RandomHorizontalFlip = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
        self.transform_normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
        if self.split in {'VAL', 'TEST'}:
            image = Image.open(loadpath)
            img = image.resize((256, 256), Image.ANTIALIAS)
            img = np.asarray(img)
            # print(img.shape)
            img = img.transpose(2, 0, 1)
            all_feats = torch.Tensor(img).unsqueeze(0)           
            self.all_feats=all_feats
            self.cpi = 1
            # Load encoded captions (completely into memory)
            file_name_val = []

    def __getitem__(self, i):
        all_feats =torch.FloatTensor(self.all_feats[i]/ 255.)
        return all_feats

    def __len__(self):
        return 1

def get_dataloader(data_name, split, workers, batch_size,word2idx,folder):
    dataset = CaptionDataset(data_name, split, data_folder=folder)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True,\
            collate_fn = create_collate_fn(word2idx))
    return dataloader