import config
from dataloader import iMetDataset
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import copy
import utils

from engine import train_one_epoch, evaluate


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    conf = config.Config()
    train_df = pd.read_csv(conf.DATA_PATH+'/train.csv')
    print('Loaded train_df!')
    print(train_df.head())

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=46, pretrained_backbone=True)
    print('Loaded model!')

    d = iMetDataset(conf, train_df)
    data_loader = torch.utils.data.DataLoader(d, batch_size=8, shuffle=True, num_workers=4)
    print('Initialized data loader')

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
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
