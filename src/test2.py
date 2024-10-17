import torch

outputs = torch.tensor([
    [0.1, 0.2],
    [0.05, 0.4]
])

print(outputs.argmax(0))  # 按行比较，返回每一列的最大值的索引
print(outputs.argmax(1)) # 按列比较，返回每一行的最大值的索引

preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print((preds == targets).sum())