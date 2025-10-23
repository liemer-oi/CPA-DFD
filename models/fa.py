import logging
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import FCSNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy, accuracy
from utils.sampler import GaussianSampler

EPSILON = 1e-8


class FA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self.size = self.args["size"]
        self.sampler = GaussianSampler(self._device)
    

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        self.save_checkpoint("checkpoint/{}/{}/{}/{}".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"]))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes*4)
        if self._cur_task > 0:
            self._network.update_transfer()

        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        
        resume = False
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("checkpoint/{}/{}/{}/{}/phase{}.pkl".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
            logging.info('!!!resume!!!')
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:
            if self._cur_task == 0:
                self._epoch_num = self.args["init_epochs"]
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"], weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["init_epochs"])
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                self._epoch_num = self.args["epochs"]
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
                self._update_train(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                cov = np.cov(vectors.T).astype(np.float32)
                self.sampler.add_class(class_mean, cov)


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                features = self._network_module_ptr.extract_vector(inputs)
                logits = self._network_module_ptr.fc(features)["logits"]
                loss_clf = F.cross_entropy(logits/self.args["temp"], aug_targets)
                aug_logits = 0
                for i in range(1, 4):
                    aug_logits = aug_logits + logits[i::4, i::4] / 3
                non_aug_logits = logits[0::4, 0::4]
                self_distillation_loss = F.kl_div(F.log_softmax(non_aug_logits, 1),
                                    F.softmax(aug_logits.detach(), 1),
                                    reduction='batchmean')
                loss = loss_clf + self_distillation_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aug_targets.expand_as(preds)).cpu().sum()
                total += len(aug_targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            train_num = len(train_loader)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/train_num, train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/train_num, train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
            

    def _update_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_cls, losses_sd, losses_dfd, losses_proto = 0., 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                logits, loss_cls, loss_sd, loss_dfd, loss_proto = self._compute_loss(inputs, targets, aug_targets)
                loss = loss_cls + loss_dfd + loss_proto + loss_sd 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_cls += loss_cls.item()
                losses_sd += loss_sd.item()
                losses_dfd += loss_dfd.item()
                losses_proto += loss_proto.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aug_targets.expand_as(preds)).cpu().sum()
                total += len(aug_targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            train_num = len(train_loader)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/train_num, train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_cls {:.3f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/train_num, train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)



    def _compute_loss(self, inputs, targets, aug_targets):
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_cls = F.cross_entropy(logits/self.args["temp"], aug_targets)

        aug_logits = 0
        for i in range(1, 4):
            aug_logits = aug_logits + logits[i::4, i::4] / 3
        non_aug_logits = logits[0::4, 0::4]
        self_distillation_loss = F.kl_div(F.log_softmax(non_aug_logits, 1),
                            F.softmax(aug_logits.detach(), 1),
                            reduction='batchmean')
        
        features_old = self.old_network_module_ptr(inputs)["features"]
        loss_dfd = dfd_loss(feat_old=features_old, feat_new=features)
        
        proto_features, proto_targets = self.sampler.sample(n_samples=self.args["batch_size"])
        proto_features = proto_features.to(self._device,non_blocking=True)
        proto_targets = proto_targets.to(self._device,non_blocking=True)

        # proto_features = proto_features.to(self._device,non_blocking=True)
        # proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(self._device,non_blocking=True)
        
        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        
        loss_proto = self.args["lambda"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets*4)
                
        return logits, loss_cls, self_distillation_loss, loss_dfd, loss_proto
    
    
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:,::4]
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
                outputs = self._network(inputs)["logits"][:,::4]
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
    

def dfd_loss(feat_old, feat_new):
    eps = 1e-6
    if feat_old.shape != feat_new.shape:
        raise ValueError(f"feat_old and feat_new must have same shape")
    ft_norm = feat_old.norm(p=2, dim=1).clamp_min(eps)
    fs_norm = feat_new.norm(p=2, dim=1).clamp_min(eps)
    mag_diff_sq = (ft_norm - fs_norm).pow(2)
    dot = (feat_old * feat_new).sum(dim=1) 
    cos_sim = dot / (ft_norm * fs_norm).clamp_min(eps)  
    ang_term = 2.0 * (1.0 - cos_sim)
    loss_per_sample = mag_diff_sq + ang_term
    loss_per_sample = loss_per_sample.clamp_min(0.0)
    return loss_per_sample.sum()
    