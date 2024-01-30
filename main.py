import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import BertModel
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split


class MultimodalDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, tokenizer=None,is_test=False):
       
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.is_test = is_test  # 判断是否为测试集

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 读取图片
        img_name = os.path.join(self.data_dir, f"{self.annotations.iloc[idx, 0]}.jpg")
        image = Image.open(img_name)

        # 对图片应用转换
        if self.transform:
            image = self.transform(image)

        # 处理文本数据
        txt_name = os.path.join(self.data_dir, f"{self.annotations.iloc[idx, 0]}.txt")
        with open(txt_name, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")

        # 获取标签
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if not self.is_test:
            label_str = self.annotations.iloc[idx, 1]
            label = label_map[label_str]
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)  # 对于测试集，可以返回一个默认值

        return image, text, label
    
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel, self).__init__()
        # 图像模型 ResNet-18
        self.image_model = resnet18(pretrained=True)
        # 替换ResNet的分类层
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        # 文本模型 BERT
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        # 添加批量归一化层
        self.image_bn = nn.BatchNorm1d(num_ftrs)
        self.text_bn = nn.BatchNorm1d(self.text_model.config.hidden_size)
        

        # 特征融合和分类层
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + self.text_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)  
        )

    def forward(self, images, input_ids, attention_mask):
        # 处理图像
        image_features = self.image_model(images)
        
        # 处理文本
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.pooler_output
        text_features = self.text_bn(text_features)

        # 融合特征
        combined_features = torch.cat((image_features, text_features), dim=1)

        # 分类
        outputs = self.classifier(combined_features)
        return outputs

# 图像的转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 添加随机水平翻转
    transforms.RandomRotation(10),      # 添加随机旋转
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = MultimodalDataset(annotations_file='train.txt', data_dir='data', transform=transform)

val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 初始化模型
model = MultimodalSentimentModel() 
model = model.to(device)  

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
# 定义梯度截断阈值
max_grad_norm = 1.0

# 训练循环
num_epochs = 5  
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, texts, labels in train_loader:
        # 提取文本数据中的输入
        input_ids = texts['input_ids'].squeeze(1).to(device)
        attention_mask = texts['attention_mask'].squeeze(1).to(device)

       # 传入GPU
        images = images.to(device)
        labels = labels.to(device)

        # 清除优化器的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()

        # 梯度截断
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        running_loss += loss.item()
        # 验证步骤
    model.eval()  # 评估模式
    correct = 0
    total = 0
    with torch.no_grad():  
        for images, texts, labels in val_loader:
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Validation Accuracy: {accuracy}%')


# 读取测试集
custom_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=256)
test_dataset = MultimodalDataset(annotations_file='test_without_label.txt', data_dir='data', transform=transform, tokenizer=custom_tokenizer,is_test=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# 模型预测
model.eval()  # 评估模式
predictions = []
with torch.no_grad():
    for images, texts, _ in test_loader:
        input_ids = texts['input_ids'].squeeze(1).to(device)
        attention_mask = texts['attention_mask'].squeeze(1).to(device)
        images = images.to(device)

        outputs = model(images, input_ids, attention_mask)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

# 将预测标签映射回原始标签
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predictions = [label_map[p] for p in predictions]

# 读取测试集文件并替换标签
with open('test_without_label.txt', 'r') as file:
    lines = file.readlines()

# 确保第一行保持不变
lines[0] = 'guid,tag\n'

# 替换其余行的标签
for i, prediction in enumerate(predictions, start=1):
    lines[i] = lines[i].split(',')[0] + ',' + prediction + '\n'

# 写入结果
with open('result.txt', 'w') as file:
    file.writelines(lines)
                    