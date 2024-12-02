from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import sys

def createDataloader(
        train_path: str,
        test_path: str,
        batch_size: int,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
):
    """
    this Funtion will take image path and return a torch dataloader
    Args:
        train_path: /path/to/img
        test_path: /path/to/img
        batch_size: num of sample per epoch
        train_transform: Transform for train
        test_transform: Transform for test
    Returns:
        Return a train data loader and test data loader
    """

    train_data = datasets.ImageFolder(train_path, transform=train_transform, target_transform=None)
    test_data = datasets.ImageFolder(test_path, transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, train_data.classes

# if __name__ == "__main__":
    # print(sys.argv)
    # train_path = sys.argv[1]
    # test_path = sys.argv[2]
    # batch_size = sys.argv[3]
    
    # train_transform = transforms.Compose(
    #     [
    #         transforms.Resize((224,224)),
    #         transforms.TrivialAugmentWide(num_magnitude_bins=31),
    #         transforms.ToTensor(),
    #     ]
    # ) 

    # test_transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ]
    # ) 
