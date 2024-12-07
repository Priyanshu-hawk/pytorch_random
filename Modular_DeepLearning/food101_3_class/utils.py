from torch import nn
import torch
from torchinfo import summary
from tqdm import tqdm
import copy
import os
import time
import json


def loss_fn(device):
    return nn.CrossEntropyLoss().to(device)

def optimizer( optimz: str, learning_rate: float, model: nn.Module):
    """
    rmsprop
    
    """
    optim_funcs = {
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
        "sdg": torch.optim.SGD,
    }
    return optim_funcs[optimz](params=model.parameters(), lr=learning_rate)

def acc_func(y_preds, y):
    # print(y_preds)
    acc_score = torch.eq(y_preds, y).sum() / len(y)
    return acc_score


def trainit(model: nn.Module, 
            train_dataloader, 
            test_dataloader, 
            loss_fn: nn.modules.loss , 
            optimizer: torch.optim, 
            epochs: int, 
            cala_test_loss_acc: bool, 
            device: torch.device):

    
    def train(model: nn.Module, loss_fn: nn.modules.loss, optim, train_dataloader):
        train_acc = 0
        train_loss = 0
        model.train()
        for x, y in train_dataloader:
            x=x.to(device)
            y=y.to(device)
            y_logits = model(x)
            loss = loss_fn(y_logits, y)
            train_loss+=loss.item()
            train_acc+=acc_func(torch.argmax(y_logits, dim=1), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        return train_loss/len(train_dataloader), train_acc/len(train_dataloader)
        
    def test(model: nn.Module, loss_fn: nn.modules.loss, test_dataloader):
        model.eval()
        test_loss = 0
        test_acc = 0
        for x, y in test_dataloader:
            x=x.to(device)
            y=y.to(device)
            y_logits = model(x)
            loss = loss_fn(y_logits, y)
            test_loss+=loss
            test_acc += acc_func(torch.argmax(y_logits, dim=1), y)
            loss.backward()
        
        return test_loss/len(test_dataloader) , test_acc/len(test_dataloader)
        

    best_loss = float('inf')
    best_model_weights = None
    patience = 10

    train_data = {
        "Acc": [],
        "Loss": []
    }

    test_data = {
        "Acc": [],
        "Loss": []
    }

    for e in tqdm(range(epochs)):
        train_loss, train_acc = train(model, loss_fn, optimizer, train_dataloader)
        print("Train: Loss - {} | Accuracy - {}".format(train_loss, train_acc))
        
        train_data["Acc"].append(train_acc)
        train_data["Loss"].append(train_loss)

        if cala_test_loss_acc:
            test_loss, test_acc = test(model, loss_fn, test_dataloader)
            print("Test: Loss - {} | Accuracy - {}".format(test_loss, test_acc))

            test_data["Acc"].append(test_acc)
            test_data["Loss"].append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
                patience = 10  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break
    
    run_save_time = int(time.time())
    LOGS_PATH = "logs/{}".format(run_save_time)
    MODEL_PATH = "model/{}".format(run_save_time) 
    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    torch.save(best_model_weights, os.path.join(MODEL_PATH, "best_model.pth"))

    with open(os.path.join(LOGS_PATH, "train.json"), "w") as f:
        json.dump({k: [float(i) for i in v] for k, v in train_data.items()}, f)
    
    with open(os.path.join(LOGS_PATH, "test.json"), "w") as f:
        json.dump({k: [float(i) for i in v] for k, v in test_data.items()}, f)
