import numpy as np
import cv2
import pandas as pd
import json
import os
import numpy as np
import torch
from PIL import Image


class iMetDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, transforms,random_seed=0):       # transforms,
        self.CONFIG = config
        self.TRAINING_DATA_PATH = self.CONFIG.DATA_PATH + '/train/'
        self.TRAINING_DATA_FRAME = df
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = self.TRAINING_DATA_FRAME.ImageId.unique()
        np.random.seed(random_seed)

    def make_single_mask(self, encoded_string, height, width):
        splitted_string = np.array(list(map(int, encoded_string.split()))).reshape(-1,2)
        mask = np.zeros((height*width), dtype=np.uint8)
        for start_indice, run_length in splitted_string:
            start_indice-=1
            mask[start_indice:start_indice+run_length] = 1
        return mask.reshape((height, width), order='F')

    def get_box(self, mask):
        w = np.sum(mask, axis=0)
        h = np.sum(mask, axis=1)
        x1, x2 = 0, len(w)-1
        y1, y2 = 0, len(h)-1
        while w[x1]==0:
            x1+=1
        while w[x2]==0:
            x2-=1
        while h[y1]==0:
            y1+=1
        while h[y2]==0:
            y2-=1
        return np.array([x1, y1, x2, y2])

    def get_ground_truth(self, image_id):
        query = self.TRAINING_DATA_FRAME[self.TRAINING_DATA_FRAME.ImageId==image_id]
        encoded_pixels = query.EncodedPixels
        class_ids = query.ClassId
        height, width = list(query.Height)[0], list(query.Width)[0]

        masks=[]
        labels=[]
        boxes=[]

        for _, (encoded_pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            sub_mask = self.make_single_mask(encoded_pixel_str, height, width)
            masks.append(sub_mask)
            boxes.append(self.get_box(sub_mask))
            labels.append(class_id)

        return {'boxes':torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'masks': torch.as_tensor(masks, dtype=torch.uint8)}


    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.TRAINING_DATA_PATH + self.imgs[idx] + '.jpg'
        img = np.array(cv2.cvtColor( cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255)
        img = np.moveaxis(img, -1, 0)
        print(img.shape)
        img = torch.tensor(img, dtype=torch.float32)

        target = self.get_ground_truth(self.imgs[idx])

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((target['labels'].shape[0],), dtype=torch.int64)
        boxes = target['boxes']
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        print(img.shape, img_path, idx)

        #target["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd

        # if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        #print("Shape of target[boxes,image_id,masks,labels,area,iscrowd]: "+str(target['boxes'].shape)+str(target['image_id'].shape)
               # +str(target['masks'].shape)+str(target['labels'].shape)+str(target['area'].shape)+','+str(target['iscrowd'].shape))

        return img, target

    def __len__(self):
        return len(self.imgs)

class Dataset_for_pytorch():
    def __init__ (self, config, df, random_seed=0):
        self.CONFIG = config
        self.TRAINING_DATA_PATH = self.CONFIG.DATA_PATH + '/train/'
        self.TRAINING_DATA_FRAME = df
        self.DATASET_IDXS = self.TRAINING_DATA_FRAME.ImageId.unique()
        self.DATASET_SIZE = self.DATASET_IDXS.shape[0]

        np.random.seed(random_seed)
        np.random.shuffle(self.DATASET_IDXS)
        self.CURRENT_IDX=0

    def get_image(self, image_id, resize=True):
        img = cv2.cvtColor( cv2.imread(self.TRAINING_DATA_PATH+image_id+'.jpg'), cv2.COLOR_BGR2RGB)
        if resize:
            return cv2.resize(img,
                              (self.CONFIG.IMAGE_SIZE, self.CONFIG.IMAGE_SIZE),
                              interpolation = cv2.INTER_NEAREST)/255
        else:
            return img/255

    def make_single_mask(self, encoded_string, height, width):
        splitted_string = np.array(list(map(int, encoded_string.split()))).reshape(-1,2)
        mask = np.zeros((height*width), dtype=np.uint8)
        for start_indice, run_length in splitted_string:
            start_indice-=1
            mask[start_indice:start_indice+run_length] = 1
        return mask.reshape((height, width), order='F')

    def get_box(self, mask):
        w = np.sum(mask, axis=0)
        h = np.sum(mask, axis=1)
        x1, x2 = 0, len(w)-1
        y1, y2 = 0, len(h)-1
        while w[x1]==0:
            x1+=1
        while w[x2]==0:
            x2-=1
        while h[y1]==0:
            y1+=1
        while h[y2]==0:
            y2-=1

        return np.array([x1, y1, x2, y2])

    def get_ground_truth(self, image_id, resize=True):

        query = self.TRAINING_DATA_FRAME[self.TRAINING_DATA_FRAME.ImageId==image_id]
        encoded_pixels = query.EncodedPixels
        class_ids = query.ClassId
        height, width = list(query.Height)[0], list(query.Width)[0]

        masks=[]
        labels=[]
        boxes=[]

        for _, (encoded_pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            sub_mask = self.make_single_mask(encoded_pixel_str, height, width)
            if resize:
                sub_mask = cv2.resize(sub_mask,
                                      (self.CONFIG.OUTPUT_MASK_SIZE, self.CONFIG.OUTPUT_MASK_SIZE),
                                      interpolation=cv2.INTER_NEAREST)
            masks.append(sub_mask)
            boxes.append(self.get_box(sub_mask))
            labels.append(class_id)

        return {'boxes':np.array(boxes),
                'labels': np.array(labels),
                'masks': np.array(masks)}


    def sample_next_batch(self):
        next_idx = self.CURRENT_IDX+self.CONFIG.BATCH_SIZE
        if next_idx>self.DATASET_SIZE-1:
            next_idx = self.CONFIG.BATCH_SIZE-(self.DATASET_SIZE-self.CURRENT_IDX)
            batch_idxs = np.concatenate((self.DATASET_IDXS[self.CURRENT_IDX:],
                                        self.DATASET_IDXS[:next_idx]))
            self.CURRENT_IDX = next_idx
            return batch_idxs, True
        else:
            batch_idxs = self.DATASET_IDXS[self.CURRENT_IDX:next_idx]
            self.CURRENT_IDX = next_idx
            return batch_idxs, False

    def get_next_batch(self, resize=True):
        batch_idxs, epoch_finish = self.sample_next_batch()
        print(batch_idxs)
        train_images=[]
        ground_truth=[]
        for image_id in batch_idxs:
            train_images.append(self.get_image(image_id, resize=resize))
            ground_truth.append(self.get_ground_truth(image_id, resize=resize))
        return train_images, ground_truth, epoch_finish

    def get_label_dictionaries(self):
        json_file = json.loads(open(self.CONFIG.DATA_PATH+'/label_descriptions.json', 'r').read())
        categories = {id_:category
                      for (id_,category)
                      in zip([i['id'] for i in json_file['categories']],
                             [n['name'] for n in json_file['categories']])}
        attributes = {id_:category
                      for (id_,category)
                      in zip([i['id'] for i in json_file['attributes']],
                             [n['name'] for n in json_file['attributes']])}
        return categories, attributes
