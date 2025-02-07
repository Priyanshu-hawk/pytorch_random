import torch
import torch.nn as nn
from dataset import get_dataloader, get_label_model, download_dataset
from model import UNetCore
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds_path = "../dataset/cityscapes_images/"

if not os.path.exists(ds_path): #noqa
    os.makedirs(ds_path)
    url = "https://www.kaggle.com/api/v1/datasets/download/dansbecker/cityscapes-image-pairs"
    filename = ds_path + "cityscapes-image-pairs.zip"
    download_dataset(url, filename)

    #unzip the dataset
    import zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(ds_path)


dataset_path = os.path.join(ds_path, "cityscapes_data/")
train_path = os.path.join(dataset_path, "train/")
val_path = os.path.join(dataset_path, "val/")

# HPrams
epochs = 60
lr = 0.001
ncluster = 10
batch_size = 32

label_model = get_label_model(n_clusters=ncluster)
dataloader = get_dataloader(train_path, label_model, batch_size=batch_size, shuffle=True)

print("DataLoader created")

model = UNetCore(3, 10).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx, (data, target) in loop:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = loss_fn(output, target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % 10 == 0:
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "model.pth")
