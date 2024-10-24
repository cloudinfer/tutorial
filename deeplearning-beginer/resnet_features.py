import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms


# 用于遍历图像
def images_iterator(image_dir):
    dataset_images = glob(f"{image_dir}/**/*.jpeg", recursive=True)
    for image_path in dataset_images:
        file_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # Resnet在预训练使用的参数
        imagenet_mean=[0.485, 0.456, 0.406]
        imagenet_std=[0.229, 0.224, 0.225]
        # 参照Resnet预训练时的图像处理方式处理
        transform = transforms.Compose([
             # 转换为 Tensor，自动将像素值归一化到 [0, 1]
            transforms.ToTensor(),  
            # 调整大小                       
            transforms.Resize((224, 224)),     
             # resnet输入要求通道数量为3             
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.size(0)==1 else x),   
            # 对像素取值归一化
            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std)                           
        ])
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)

        # dimensions of input_tensor are [1, 3, 224, 224]
        # 返回文件名和预处理之后的图像
        yield file_name, input_tensor

def pca_plot(csv_path="./test.csv"):
    # 加载提出的特征
    df_data = pd.read_csv(csv_path)
    df_x = df_data.iloc[:, 2:].to_numpy()
    df_y = df_data.iloc[:, 1].to_numpy()
    # 通过PCA算法将Resnet提取的2048个特征降为3个
    pca = PCA(n_components=3)
    x = pca.fit(df_x).transform(df_x)

    # 绘制图像的一些参数设置
    category_names = ["NORMAL", "PNEUMONIA"]
    ax = plt.figure().add_subplot(projection='3d')
    colors = ["navy", "turquoise"]
    lw = 2
    # 绘制图像
    for color, target_name in zip(colors, category_names):
        ax.scatter(
            x[df_y == target_name, 0], 
            x[df_y == target_name, 1], 
            x[df_y == target_name, 2], 
            color=color, alpha=0.8, 
            lw=lw, 
            label=target_name
        )
    plt.title("PCA of Chest X-ray")
    plt.show()


def main():
    # 加载预训练的resnet50模型
    resnet50_weight = ResNet50_Weights.DEFAULT
    print(resnet50_weight.transforms())
    resnet50_mdl = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    # 输出每一层的名称
    nodes_name, _ = get_graph_node_names(resnet50_mdl)
    # 根据输出结果，我们可以知道resnet50在通过全连接层分类之前的网络层名称是flatten
    print(nodes_name)       
    return_nodes = {
        "flatten": "final_feature_map"
    }
    # 基于选择的输出和resnet50构建特征提取器
    feature_extracter = create_feature_extractor(resnet50_mdl, return_nodes=return_nodes)
    # 设定影像路径
    chest_xray = os.path.join(os.getcwd(), "chest_xray")
    train_dataset_dir = os.path.join(chest_xray, "train")
    test_dataset_dir = os.path.join(chest_xray, "test")
    # 设定列名
    column_names = ["patient_id", "category"] + [f"resnet_feature_{i+1}" for i in range(2048)]
    
    with torch.no_grad():
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        # 进行训练集印象特征提取
        for image_name, image_tensor in tqdm(images_iterator(train_dataset_dir)):
            # dimensions of out is [1, 2048]
            out = feature_extracter(image_tensor)
            out_features = out["final_feature_map"]
            out_features = out_features.cpu().numpy()[0]
            if "NORMAL" in image_name:
                category = "NORMAL"
            else:
                category = "PNEUMONIA"
            row_data = [image_name, category] + list(out_features)
            df_train = pd.concat([df_train, pd.Series(row_data)], ignore_index=True, axis=1)
        # 保存到csv文件
        df_train = df_train.T
        df_train.columns = column_names
        df_train.to_csv("train.csv", index=False)
        # 进行测试集影像特征提取
        for image_name, image_tensor in tqdm(images_iterator(test_dataset_dir)):
            # dimensions of out is [1, 2048]
            out = feature_extracter(image_tensor)
            out_features = out["final_feature_map"]
            out_features = out_features.cpu().numpy()[0]
            if "NORMAL" in image_name:
                category = "NORMAL"
            else:
                category = "PNEUMONIA"
            row_data = [image_name, category] + list(out_features)
            df_test = pd.concat([df_test, pd.Series(row_data)], ignore_index=True, axis=1)
        # 保存到csv文件
        df_test = df_test.T
        df_test.columns = column_names
        df_test.to_csv("test.csv", index=False)
    

if __name__ == "__main__":
    main()
    pca_plot()