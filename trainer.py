import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, average_forgetting
import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import sklearn
from torch.utils.data import DataLoader

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "log/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    checkpoint_dir = "checkpoint/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logfilename = "log/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_all_acc, nme_all_acc = [], []
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy, cnn_acc_per_task, nme_acc_per_task = model.eval_task()
        model.after_task()

        cnn_acc_per_task.extend((data_manager.nb_tasks-1-task)*[0])
        cnn_all_acc.append(cnn_acc_per_task)
        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN average incremental accuracy: {}".format(np.mean(cnn_curve["top1"])))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        if task == data_manager.nb_tasks-1:
            cnn_forgetting = average_forgetting(np.array(cnn_all_acc), data_manager.nb_tasks)
            logging.info("CNN average forgetting: {}".format(cnn_forgetting))

        if nme_accy is not None:
            nme_acc_per_task.extend((data_manager.nb_tasks-1-task)*[0])
            nme_all_acc.append(nme_acc_per_task)
            logging.info("NME: {}".format(nme_accy["grouped"]))

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME average incremental accuracy: {}".format(np.mean(nme_curve["top1"])))
            logging.info("NME top5 curve: {}".format(nme_curve["top5"]))
            if task == data_manager.nb_tasks-1:
                nme_forgetting = average_forgetting(np.array(nme_all_acc), data_manager.nb_tasks)
                logging.info("NME average forgetting: {}".format(nme_forgetting))
        else:
            logging.info("No NME accuracy.\n")
        

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
