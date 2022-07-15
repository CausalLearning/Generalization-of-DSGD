import datetime
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from models import get_model
from dataset import get_dataset
from gpu_work import Worker
from config import get_config

TEST_ACCURACY = 0


def work():
    config = get_config()
    print(config)
    run(**config)


def run(num_epoch, model_name, dataset_name, mode, size,
        batch_size, lr, momentum, weight_decay,
        milestones, gamma, gpu, early_stop, seed, **kwargs):
    tb = SummaryWriter(comment=f"{seed}_{model_name}_{dataset_name}_{mode}_{batch_size}_{lr}_{size}")

    temp_train_loader, temp_test_loader, input_size, classes = get_dataset(rank=0,
                                                                           dataset_name=dataset_name,
                                                                           split=None,
                                                                           batch_size=256,
                                                                           is_distribute=False,
                                                                           seed=seed,
                                                                           **kwargs)
    temp_model = get_model(model_name, input_size, classes)
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    P = generate_P(mode, size)
    criterion = nn.CrossEntropyLoss()
    num_step = temp_train_loader.__len__() * 64
    worker_list = []
    for rank in range(size):
        split = [1.0 / size for _ in range(size)]
        train_loader, test_loader, input_size, classes = get_dataset(rank=rank,
                                                                     dataset_name=dataset_name,
                                                                     split=split,
                                                                     batch_size=batch_size,
                                                                     seed=seed,
                                                                     **kwargs)
        torch.manual_seed(rank)
        model = get_model(model_name, input_size, classes)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        schedule = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        worker = Worker(rank=rank, model=model,
                        train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, schedule=schedule, gpu=gpu)
        worker_list.append(worker)
        if train_loader.__len__() < num_step:
            num_step = train_loader.__len__()

    print(f"| num_step: {num_step}")

    total_step = 0
    for epoch in range(1, num_epoch + 1):
        start = datetime.datetime.now()
        for worker in worker_list:
            worker.update_iter()

        for step in range(num_step):
            total_step += 1

            weight_dict_list = []
           
            for worker in worker_list:
                weight_dict_list.append(worker.model.state_dict())
                worker.step()
                
            for worker in worker_list:
                for name, param in worker.model.named_parameters():
                    param.data = torch.zeros_like(param.data)
                    for i in range(size):
                        p = P[worker.rank][i]
                        param.data += weight_dict_list[i][name].data * p
                worker.update_grad()

            temp_model = copy.deepcopy(worker_list[0].model)
            for name, param in temp_model.named_parameters():
                for worker in worker_list[1:]:
                    param.data += worker.model.state_dict()[name].data
                param.data /= size

            if total_step % 50 == 0:
                test_all(temp_model, temp_train_loader, temp_test_loader,
                         criterion, None, total_step, tb, device, n_swap=kwargs.get('n_swap'))
            if total_step == early_stop:
                break
            
            end = datetime.datetime.now()
            print(f"\r| Train | epoch: {epoch}|{num_epoch}, step: {step}|{num_step}, time: {(end - start).seconds}s",
                  flush=True, end="")
        if total_step == early_stop:
            break

def test_all(model, train_loader, test_loader, criterion, epoch, total_step, tb, device, n_swap=None):
    print(f"\n| Test All |", flush=True, end="")
    model.to(device)
    model.eval()
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total
    if epoch is None:
        print(f'\n| Test All Train Set |'
              f' total step: {total_step},'
              f' loss: {total_train_loss:.4},'
              f' acc: {total_train_acc:.4%}', flush=True)
    else:
        print(f'\n| Test All Train Set |'
              f' epoch: {epoch},'
              f' loss: {total_train_loss:.4},'
              f' acc: {total_train_acc:.4%}', flush=True)

    total_loss, total_correct, total, step = 0, 0, 0, 0
    for batch in test_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_test_loss = total_loss / step
    total_test_acc = total_correct / total
    if epoch is None:
        print(f'\n| Test All Test Set |'
              f' total step: {total_step},'
              f' loss: {total_test_loss:.4},'
              f' acc: {total_test_acc:.4%}', flush=True)
    else:
        print(f'\n| Test All Test Set |'
              f' epoch: {epoch},'
              f' loss: {total_test_loss:.4},'
              f' acc: {total_test_acc:.4%}', flush=True)

    if epoch is None:
        tb.add_scalar("test loss - train loss", total_test_loss - total_train_loss, total_step)
        tb.add_scalar("test loss", total_test_loss, total_step)
        tb.add_scalar("train loss", total_train_loss, total_step)
        tb.add_scalar("test acc", total_test_acc, total_step)
        tb.add_scalar("train acc", total_train_acc, total_step)
    else:
        tb.add_scalar("test loss - train loss", total_test_loss - total_train_loss, epoch)
        tb.add_scalar("test loss", total_test_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("test acc", total_test_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    if n_swap is not None:
        if total_test_acc > TEST_ACCURACY:
            torch.save(model.state_dict(), f"./trained/resnet18_tinyimagenet_{n_swap}_best.pt")
        torch.save(model.state_dict(), f"./trained/resnet18_tinyimagenet_{n_swap}_last.pt")


def generate_P(mode, size):
    result = torch.zeros((size, size))
    if mode == "all":
        result = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            result[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            result[i][i] = 1 / 3
            result[i][(i - 1 + size) % size] = 1 / 3
            result[i][(i + 1) % size] = 1 / 3
    elif mode == "star":
        for i in range(size):
            result[i][i] = 1 - 1 / size
            result[0][i] = 1 / size
            result[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(len(topo_neighbor_with_self[i]),
                                           len(topo_neighbor_with_self[j]))
            topo[i][i] = 2.0 - topo[i].sum()
        result = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        result = torch.tensor(topo, dtype=torch.float)
    print(result, flush=True)
    return result


if __name__ == '__main__':
    work()
