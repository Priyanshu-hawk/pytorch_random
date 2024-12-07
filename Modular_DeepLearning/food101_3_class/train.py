from .model import ResidualBlockResNet50Above, ResNet50Above
from .data_setup import createDataloader
from .utils import loss_fn, optimizer, trainit
from torchvision import transforms
import torch

import sys

if __name__ == "__main__":
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    batch_size = sys.argv[3]
    epochs = sys.argv[4]

    train_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ]
    ) 

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    ) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, class_names = createDataloader(train_path=train_data_path, 
                     test_path=test_data_path, 
                     BTACH_SIZE=int(batch_size), 
                     train_transform=train_transform,
                     test_transform=test_transform)
    
    _model = ResNet50Above(ResidualBlockResNet50Above, [3,4,23,3], len(class_names))
    _model = _model.to(device)

    trainit(_model, train_loader, test_loader, loss_fn(device), optimizer("adam", 0.001, _model), int(epochs), cala_test_loss_acc=True, device=device)
    


