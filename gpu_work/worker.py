from typing import List

from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn

import torch


class Worker:
    def __init__(self, rank, model: Module,
                 train_loader: DataLoader, test_loader: DataLoader,
                 optimizer, schedule,
                 gpu=True):
        self.rank = rank
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loader_iter = train_loader.__iter__()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.schedule = schedule
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.model.to(self.device)

    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()
        self.model.to(self.device)

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        # self.model.cpu()

    def step_csgd(self):
        self.model.train()
        self.model.to(self.device)

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        # self.model.cpu()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        self.optimizer.step()
        self.schedule.step()

    def schedule_step(self):
        self.schedule.step()
