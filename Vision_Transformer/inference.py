from model import ResidualBlockResNet50Above, ResNet50Above
import sys
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
from torchinfo import summary

## hardcoding the class names
class_names = ['pizza', 'steak', 'sushi']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = ResNet50Above(ResidualBlockResNet50Above, [3,4,23,3], len(class_names))
    # print(summary(model, input_size=(32, 3, 224, 224)))
    
    model.load_state_dict(torch.load(sys.argv[1]))

    img_path = sys.argv[2]
    img = Image.open(img_path)
    img = transforms.ToTensor()(img).unsqueeze(0)

    img = img.to(device)

    inference = model(img)
    print(inference)
    print(class_names[torch.argmax(inference).item()])

