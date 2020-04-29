import sys
from collections import defaultdict
import csv
import os
import gc
import sys
import json
import random
from pathlib import Path

import cv2 # CV2 for image manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from util import create_mask, plot_segmented_images



def main(args):
    with open('./data/label_descriptions.json', 'r') as file:
        label_desc = json.load(file)

    train_df = pd.read_csv('./data/train_csv/train.csv')
    print(train_df.head())

    categories_df = pd.DataFrame(label_desc.get('categories'))
    attributes_df = pd.DataFrame(label_desc.get('attributes'))
    print(f'# of categories: {len(categories_df)}') #46
    print(f'# of attributes: {len(attributes_df)}') #294

    plot_segmented_images(train_df)

if __name__ == '__main__':
    main(sys.argv)
