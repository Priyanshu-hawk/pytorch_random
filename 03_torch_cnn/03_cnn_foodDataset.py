import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import gc


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
# device = torch.device("cpu")
logging.info(f"Using Device - {device}")

imgTransform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()]
)

logging.info(f"Using image transform as {imgTransform}")

BASE_FOOD_DS = os.path.join("../dataset/food-101")
print(BASE_FOOD_DS)
BASE_FOOD_IMGS = os.path.join(BASE_FOOD_DS, "images")
BASE_FOOD_META = os.path.join(BASE_FOOD_DS, "meta")
TRAIN_JSON_PATH = os.path.join(BASE_FOOD_META, "train.json")
TEST_JSON_PATH = os.path.join(BASE_FOOD_META, "test.json")
CLASSES_TEXT = os.path.join(BASE_FOOD_META, "classes.txt")

TRAIN_JSON = json.load(open(TRAIN_JSON_PATH))
TEST_JSON = json.load(open(TEST_JSON_PATH))

CLASS_NAMES = []
with open(CLASSES_TEXT, "r") as f:
    for f_p in f.readlines():
        CLASS_NAMES.append(f_p.strip("\n"))

logging.info(f"Total Classes are {len(CLASS_NAMES)}")
logging.info(f"Total No of Imgs in Train {len(TRAIN_JSON[CLASS_NAMES[0]])}")
logging.info(f"Total No of Imgs in Test {len(TEST_JSON[CLASS_NAMES[0]])}")

NEW_DS = "../dataset/food101_torch"

# run once
# creating dataset acording to pytorch format

#format = train/class_name/imgs 
#BASE_FOOD_IMGS,TRAIN_JSON[CLASS_NAMES[rand_Class_name]][random.randint(0,750)]+".jpg"



#train loop
if not os.path.exists(NEW_DS):
    os.makedirs(NEW_DS, exist_ok=True)
    for c_i , _class in enumerate(CLASS_NAMES):
        print(_class)
        class_path = os.path.join(NEW_DS, "train", _class)
        os.makedirs(class_path, exist_ok=True)
        i = 0
        for _img in range(0, len(TRAIN_JSON[CLASS_NAMES[0]])):
            img_path = os.path.join(class_path, str(i)+".jpg")
            # print(os.path.join(BASE_FOOD_IMGS,TRAIN_JSON[CLASS_NAMES[c_i][_img]]+".jpg"))
            shutil.copy(os.path.join(BASE_FOOD_IMGS,TRAIN_JSON[CLASS_NAMES[c_i]][_img]+".jpg"),  img_path)
            i+=1

    #test loop
    for c_i , _class in enumerate(CLASS_NAMES):
        print(_class)
        class_path = os.path.join(NEW_DS, "test", _class)
        os.makedirs(class_path, exist_ok=True)
        i = 0
        for _img in range(0, len(TEST_JSON[CLASS_NAMES[0]])):
            img_path = os.path.join(class_path, str(i)+".jpg")
            # print(os.path.join(BASE_FOOD_IMGS,TRAIN_JSON[CLASS_NAMES[c_i][_img]]+".jpg"))
            shutil.copy(os.path.join(BASE_FOOD_IMGS,TEST_JSON[CLASS_NAMES[c_i]][_img]+".jpg"),  img_path)
            i+=1

TRAIN_DATA_path = os.path.join(NEW_DS,"train")
TEST_DATA_path = os.path.join(NEW_DS,"test")

train_data = datasets.ImageFolder(TRAIN_DATA_path, transform=imgTransform, target_transform=None)
test_data = datasets.ImageFolder(TEST_DATA_path, transform=imgTransform, target_transform=None)

BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
            self.conv2 = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                            nn.BatchNorm2d(out_channels))
            self.downsample = downsample
            self.relu = nn.ReLU()
            self.out_channels = out_channels

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes = 10):
            super(ResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x
        
net1 = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=len(CLASS_NAMES)).to(device)

print(net1)

# sample_batch_x, sample_batch_y = next(iter(train_loader))
# sample_batch_x.shape, sample_batch_y.shape

# sample_batch_x = sample_batch_x.to(device)
# sample_batch_y = sample_batch_y.to(device)

# # net1_old = TinyNet1().to(device=device)
# with torch.inference_mode():
#     net_res = net1(sample_batch_x)

# logging.info(f"Model is on {next(net1.parameters()).device}")

# total_param = 0
# for i in net1.parameters():
#     total_param+=i.numel()
# print(total_param)

# logging.info(f"Total parameters in model {total_param}")

# def acc_fn(y_preds, y):
#     return torch.eq(y_preds, y).sum() / len(y)

# # Hyper params

# loss_fn = nn.CrossEntropyLoss().to(device)
# optim_fn = torch.optim.Adam(params=net1.parameters(), lr=0.01)

# epochs = 5
# for e in range(0, epochs):
#     total_train_loss = 0
#     total_train_acc = 0
#     for x_train, y_train in tqdm(train_loader):
#         net1.train()
#         x_train = x_train.to(device)
#         y_train = y_train.to(device)

#         y_logits = net1(x_train)
#         # print(y_preds.shape, y_train.shape)
#         train_loss = loss_fn(y_logits, y_train)
#         total_train_loss+=train_loss

#         total_train_acc+=acc_fn(torch.argmax(y_logits, dim=1), y_train)

#         optim_fn.zero_grad()
#         train_loss.backward()
#         optim_fn.step()

#         del x_train, y_train, 
#         torch.cuda.empty_cache()
#         gc.collect()


#     with torch.inference_mode():
#         total_train_loss = total_train_loss / len(train_loader)
#         total_train_acc = total_train_acc / len(train_loader)
#         print("Total Train loss: ",total_train_loss, " | Total Train Acc:",total_train_acc)