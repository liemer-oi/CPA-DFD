import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes
        
        self.save_checkpoint("checkpoint/{}/{}/{}/{}".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._epoch_num = self.args["epochs"]
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
        
        self._train_function(train_loader, test_loader, optimizer, scheduler)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits/self.args["temp"], targets)

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epoch_num,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epoch_num,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy, cnn_acc_per_task = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy, nme_acc_per_task = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            protos = list(self._protos.values())
            y_pred, y_true = self._eval_nme(self.test_loader, protos/np.linalg.norm(protos,axis=1)[:,None])
            nme_accy, nme_acc_per_task = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
            nme_acc_per_task = None

        return cnn_accy, nme_accy, cnn_acc_per_task, nme_acc_per_task
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped, acc_per_task = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret, acc_per_task