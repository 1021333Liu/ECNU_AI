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


mode = 'image'   #在这里修改模式，mode='image' 只处理图像 mode='text' 只处理文本 mode='both'都处理。

class MultimodalDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, tokenizer=None,is_test=False,mode='both'):
        # 加载标签文件
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.is_test = is_test  
        self.mode = mode  # 消融实验结果：更换模式

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if not self.is_test:
            label_str = self.annotations.iloc[idx, 1]
            label = label_map[label_str]
        else:
            label = 0  
        label = torch.tensor(label, dtype=torch.long)

        if self.mode in ['both', 'image']:
            # 加载和处理图像
            img_name = os.path.join(self.data_dir, f"{self.annotations.iloc[idx, 0]}.jpg")
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)

        if self.mode in ['both', 'text']:
            # 加载和处理文本
            txt_name = os.path.join(self.data_dir, f"{self.annotations.iloc[idx, 0]}.txt")
            with open(txt_name, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            text = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")

        if self.mode == 'both':
            return image, text, label
        elif self.mode == 'image':
            return image, label
        elif self.mode == 'text':
            return text, label
    
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel, self).__init__()
        # 图像模型 ResNet
        self.image_model = resnet18(pretrained=True)
        # 替换ResNet的分类层
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        # 文本模型 BERT
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        # 添加批量归一化层
        self.image_bn = nn.BatchNorm1d(num_ftrs)
        self.text_bn = nn.BatchNorm1d(self.text_model.config.hidden_size)
        

        # 分别为图像和文本定义分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

        self.text_classifier = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

        self.both_classifier = nn.Sequential(
            nn.Linear(num_ftrs + self.text_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    def forward(self, images=None, input_ids=None, attention_mask=None):
        if images is not None and input_ids is None:
            # 只处理图像
            image_features = self.image_model(images)
            combined_features = self.image_bn(image_features)
            outputs = self.image_classifier(combined_features)
        elif images is None and input_ids is not None:
            # 只处理文本
            text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_output.pooler_output
            combined_features = self.text_bn(text_features)
            outputs = self.text_classifier(combined_features)
        else:
            # 处理图像
            image_features = self.image_model(images)
            # 处理文本
            text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_output.pooler_output
            text_features = self.text_bn(text_features)

            # 融合特征
            combined_features = torch.cat((image_features, text_features), dim=1)
            outputs = self.both_classifier(combined_features)
        return outputs

# 图像的转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 添加随机水平翻转
    transforms.RandomRotation(10),      # 添加随机旋转
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化数据集
dataset = MultimodalDataset(annotations_file='train.txt', data_dir='data', transform=transform,mode=mode)

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
    for data in train_loader:
        # 根据模式，获取不同类型的输入数据
        if mode == 'image':
            images, labels = data
            images = images.to(device)
            outputs = model(images=images)
        elif mode == 'text':
            texts, labels = data
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:  
            images, texts, labels = data
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)

        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        running_loss += loss.item()

    # 验证步骤
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            # 同样的模式逻辑应用于验证数据
            if mode == 'image':
                images, labels = data
                images = images.to(device)
                outputs = model(images=images)
            elif mode == 'text':
                texts, labels = data
                input_ids = texts['input_ids'].squeeze(1).to(device)
                attention_mask = texts['attention_mask'].squeeze(1).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:  
                images, texts, labels = data
                images = images.to(device)
                input_ids = texts['input_ids'].squeeze(1).to(device)
                attention_mask = texts['attention_mask'].squeeze(1).to(device)
                outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)

            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Validation Accuracy: {accuracy}%')
