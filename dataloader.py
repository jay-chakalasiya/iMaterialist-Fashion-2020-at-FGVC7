import numpy as np
import cv2
import pandas as pd


class Dataset():
    
    def __init__ (self, config, df, random_seed=5):
    

        self.CONFIG = config
        self.TRAINING_DATA_PATH = self.CONFIG.DATA_PATH + '/train/'
        self.TRAINING_DATA_FRAME = df
        self.DATASET_IDXS = self.TRAINING_DATA_FRAME.ImageId.unique()
        self.DATASET_SIZE = self.DATASET_IDXS.shape[0]
        self.CURRENT_IDX=0
        np.random.seed(random_seed)
        np.random.shuffle(self.DATASET_IDXS)
        
    def get_image(self, image_id, resize=True):
        img = cv2.cvtColor( cv2.imread(self.TRAINING_DATA_PATH+image_id+'.jpg'), cv2.COLOR_BGR2RGB)
        if resize:
            return cv2.resize(img, 
                              (self.CONFIG.IMAGE_SIZE, self.CONFIG.IMAGE_SIZE), 
                              interpolation = cv2.INTER_NEAREST)
        else:
            return img
        
    def make_single_mask(self, encoded_string, height, width):
        splitted_string = np.array(list(map(int, encoded_string.split()))).reshape(-1,2)
        mask = np.zeros((height*width), dtype=np.uint8)
        for start_indice, run_length in splitted_string:
            start_indice-=1
            mask[start_indice:start_indice+run_length] = 1
        return mask.reshape((height, width), order='F')
    
    def get_mask(self, image_id, resize=True):
        query = self.TRAINING_DATA_FRAME[self.TRAINING_DATA_FRAME.ImageId==image_id]
        encoded_pixels = query.EncodedPixels
        class_ids = query.ClassId
        height, width = list(query.Height)[0], list(query.Width)[0]
        
        if resize:
            mask = np.zeros((self.CONFIG.OUTPUT_MASK_SIZE, self.CONFIG.OUTPUT_MASK_SIZE, self.CONFIG.NO_OF_CLASSES), 
                            dtype=np.uint8)
        else:
            mask = np.zeros((height, width, self.CONFIG.NO_OF_CLASSES), 
                            dtype=np.uint8)
            
        for _, (encoded_pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            sub_mask = self.make_single_mask(encoded_pixel_str, height, width)
            if resize:
                sub_mask = cv2.resize(sub_mask, 
                                      (self.CONFIG.OUTPUT_MASK_SIZE, self.CONFIG.OUTPUT_MASK_SIZE), 
                                      interpolation=cv2.INTER_NEAREST)
            mask[:,:,class_id] = sub_mask
        return mask                    
    
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
        train_item, test_item = [], []
        for image_id in batch_idxs:
            train_item.append(self.get_image(image_id, resize=resize))
            test_item.append(self.get_mask(image_id, resize=resize))
        return np.array(train_item), np.array(test_item), epoch_finish
            