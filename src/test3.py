import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./dataset/train/ants_image/0013035.jpg"

image = Image.open(image_path)
print(image)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor() ])

image = transform(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("tudui_0.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval() # dropout和batchnorm的模型有用
with torch.no_grad():  # 不进行梯度计算
    output = model(image)
print(output)
print(output.argmax(1))