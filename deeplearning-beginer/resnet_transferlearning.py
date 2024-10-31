import os
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


class ChestXRayDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            transform=None) -> None:
        self.dataset_dir = dataset_dir
        self.transform = transform
        # 获取文件夹下所有图片路径
        self.dataset_images = glob(f"{self.dataset_dir}/**/*.jpeg", recursive=True)

    # 获取数据集大小
    def __len__(self):
        return len(self.dataset_images)
    
    # 读取图像，获取类别
    def __getitem__(self, idx):
        image_path = self.dataset_images[idx]
        image_name = os.path.basename(image_path)

        image = Image.open(image_path)
        if "NORMAL" in image_name:
            category = 0
        else:
            category = 1

        if self.transform:
            image = self.transform(image)
        
        return image, category


def prepare_model():
    # 加载预训练的模型
    resnet50_weight = ResNet50_Weights.DEFAULT
    resnet50_mdl = resnet50(weights=resnet50_weight)
    # 替换模型最后的全连接层
    num_ftrs = resnet50_mdl.fc.in_features
    resnet50_mdl.fc = nn.Linear(num_ftrs, 2)

    return resnet50_mdl

def train_model():
    # 确定使用CPU还是GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    # 加载模型
    model = prepare_model()
    model = model.to(device)
    model.train()
    # 设置loss函数和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 设置训练集数据加载相关变量
    batch_size = 32
    chest_xray = os.path.join(os.getcwd(), "chest_xray")
    train_dataset_dir = os.path.join(chest_xray, "train")
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.size(0)==1 else x),   
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ChestXRayDataset(train_dataset_dir, train_transforms)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True)
    
    # 微调模型
    for epoch in range(5):
        print_batch = 50
        running_loss = 0
        running_corrects = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # 更新模型参数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 计算评价指标
            running_loss += (loss.item() * batch_size)
            running_corrects += torch.sum(preds == labels.data)
            # 输出评价指标
            if i % print_batch == (print_batch - 1):    # print every 100 mini-batches
                accuracy = running_corrects / (print_batch * batch_size)
                print(f'Epoch: {epoch + 1}, Batch: {i + 1:5d} Running Loss: {running_loss / 50:.3f} Accuracy: {accuracy:.3f}')
                running_loss = 0.0
                running_corrects = 0
        # 保存模型
        checkpoint_name = f"epoch_{epoch}.pth"
        torch.save(model, checkpoint_name)


def test_model():
    # 确定使用CPU还是GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    # 加载模型
    checkpoint_name = "epoch_4.pth"
    model = torch.load(checkpoint_name)
    model = model.to(device)
    model.eval()
    # 设置测试集加载参数
    batch_size = 32
    chest_xray = os.path.join(os.getcwd(), "chest_xray")
    test_dataset_dir = os.path.join(chest_xray, "test")
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.size(0)==1 else x),   
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = ChestXRayDataset(test_dataset_dir, test_transforms)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False)
    # 在测试集测试模型
    with torch.no_grad():
        preds_list = []
        labels_list = []

        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_list.append(preds)
            labels_list.append(labels)
        preds = torch.cat(preds_list)
        labels = torch.cat(labels_list)
        # 计算评价指标
        corrects_num = torch.sum(preds == labels.data)
        accuracy = corrects_num / labels.shape[0]
        # 输出评价指标
        print(f"Accuracy on test dataset: {accuracy:.2%}")


if __name__ == "__main__":
    train_model()
    test_model()