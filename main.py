import config
from dataloader import iMetDataset
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import copy
import utils
from utils import get_transform

from engine import train_one_epoch, evaluate
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    conf = config.Config()
    train_df = pd.read_csv(conf.DATA_PATH+'/train.csv')
    print('Loaded train_df!')
    print(train_df.head())

    d = iMetDataset(conf, train_df, get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(d, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
    print('Initialized data loader')

    # get the model using our helper function
    model = get_model_instance_segmentation(conf.NO_OF_CLASSES)
    print('Loaded model!')
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        print('Epoch '+str(epoch))
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print("Evaluating at end of epoch")
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
