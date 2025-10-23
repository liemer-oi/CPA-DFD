import os
import numpy as np
import torch
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=50, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    acc_per_task = []
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = "{}-{}".format(
            str(0).rjust(2, "0"), str(init_cls).rjust(2, "0")
        )
    all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    acc_per_task.append(all_acc[label])
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        acc_per_task.append(all_acc[label])

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc, acc_per_task


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


def average_forgetting(all_acc_per_task, task_num):
    forgetting = []
    for i in range(task_num):
        if i == 0:
            forgetting.append(0)
        else:
            fgt = 0
            for j in range(i + 1):
                fgt += (np.max(all_acc_per_task[:, j]) - all_acc_per_task[i, j])
            fgt = np.around(fgt / i, decimals=2)
            forgetting.append(fgt)

    return forgetting



def color_perm3(images, labels):
    size = images.shape[1:]
    images = torch.stack([images,
                          torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                          torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)], 1).view(-1, *size)
    aug_targets = torch.stack([labels * 3 + k for k in range(3)], 1).view(-1)
    return images.contiguous(), aug_targets

def color_perm6(images, labels):
    size = images.shape[1:]
    images = torch.stack([images,
                        torch.stack([images[:, 0, :, :], images[:, 2, :, :], images[:, 1, :, :]], 1),
                        torch.stack([images[:, 1, :, :], images[:, 0, :, :], images[:, 2, :, :]], 1),
                        torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                        torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1),
                        torch.stack([images[:, 2, :, :], images[:, 1, :, :], images[:, 0, :, :]], 1)], 1).view(-1, *size)
    aug_targets = torch.stack([labels * 6 + k for k in range(6)], 1).view(-1)
    return images.contiguous(), aug_targets

def rot(images, labels):
    size = images.shape[1:]
    aug_images =  torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    aug_targets = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)
    return aug_images, aug_targets

def color_rot(images, labels):
    size = images.shape[1:]
    out = []
    for k in range(4):
        x = torch.rot90(images, k, (2, 3))
        out.append(x)
        out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
        out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
    aug_images = torch.stack(out, 1).view(-1, *size).contiguous()
    aug_targets = torch.stack([labels * 12 + k for k in range(12)], 1).view(-1)
    return aug_images, aug_targets