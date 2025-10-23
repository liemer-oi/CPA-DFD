from cmath import phase
import imp
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image
import sklearn
import json
import argparse
from utils.inc_net import AKACosineIncrementalNet, IncrementalNet
from datetime import datetime
import matplotlib.pyplot as plt
from utils.data_manager import DataManager
from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    plt.scatter(data[:, 0], data[:, 1], marker='o', color= plt.cm.tab10(label / 10.))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='exps/praka/cifar_LFH_P20.json',
                        help='Json file of settings.')

    return parser

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )

    device_type = args["device"]
    for device in device_type:
        device = torch.device("cuda:{}".format(device))
    # network = IncrementalNet(args, False)
    network = AKACosineIncrementalNet(args, False)
    network.update_fc(160)

    # # model = PR_AKA(args, feature_extractor, task_size, device)
    # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    print("############# Test for up2now Task #############")
    test_dataset = data_manager.get_dataset(
        np.arange(2, 6), source='test', mode='test')
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=args["num_workers"])
    network.load_state_dict(torch.load("final_models/cifar100/40/3/phase0.pkl")["model_state_dict"])
    network.to(device)
    network.eval()
    correct, total = 0.0, 0.0
    for i, (_, inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            targets_total = targets.numpy()
            features = network.extract_vector(inputs)
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            result = tsne.fit_transform(features.detach().cpu().numpy())
            break
    plot_embedding(result, targets_total, 't-SNE embedding of the digits')
    labels_set = np.unique(targets_total)
    plt.show()
    plt.savefig('visualization/PRL_0.png')
       
if __name__ == "__main__":
    main()