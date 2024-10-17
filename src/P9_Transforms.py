from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过 transforms.ToTensor()去看两个问题


img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

# 1、 transforms该如何使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)   # 转换为tensor数据类型


print(tensor_img)


# 2、 为什么我们需要Tensor数据类型

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()