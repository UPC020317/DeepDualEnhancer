# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
with  open("data/layer1/enhancer.cv.txt") as f:
       enhancer_cv = f.readlines()
       enhancer_cv = [s.strip() for s in enhancer_cv]
with  open("data/layer1/enhancer.ind.txt") as f:
       enhancer_ind = f.readlines()
       enhancer_ind = [s.strip() for s in enhancer_ind]
with  open("data/layer1/non.cv.txt") as f:
       non_cv = f.readlines()
       non_cv = [s.strip() for s in non_cv]
with  open("data/layer1/non.ind.txt") as f:
       non_ind = f.readlines()
       non_ind = [s.strip() for s in non_ind]

# %%
with  open("data/layer2/strong_742.txt") as f:
       strong_742 = f.readlines()
       strong_742 = [s.strip() for s in strong_742]
with  open("data/layer2/weak_742.txt") as f:
       weak_742 = f.readlines()
       weak_742 = [s.strip() for s in weak_742]
with  open("data/layer2/strong_100.txt") as f:
       strong_100 = f.readlines()
       strong_100 = [s.strip() for s in strong_100]
with  open("data/layer2/weak_100.txt") as f:
       weak_100 = f.readlines()
       weak_100 = [s.strip() for s in weak_100]
    
len(strong_742),len(weak_742),len(strong_100),len(weak_100)

# %%
def remove_name_1(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new

# %%
def remove_name_2(data):
    data_new = []
    for i in range(1,len(data),5):
        data_new.append(data[i].upper()+data[i+1].upper()+data[i+2].upper()+data[i+3].upper())
    return data_new

# %%
enhancer_cv = remove_name_1(enhancer_cv)
non_cv = remove_name_1(non_cv)
enhancer_ind = remove_name_1(enhancer_ind)
non_ind = remove_name_1(non_ind)
print(len(enhancer_cv),len(enhancer_cv[0]))
print(len(enhancer_ind),len(enhancer_ind[0]))
print(len(non_cv),len(non_cv[0]))
print(len(non_ind),len(non_ind[0]))
train_x = np.concatenate([enhancer_cv, non_cv], axis=0)
test_x = np.concatenate([enhancer_ind, non_ind], axis=0)
#print(len(train_x),len(test_x))

# %%
strong_742 = remove_name_2(strong_742)
weak_742 = remove_name_2(weak_742)
strong_100 = remove_name_2(strong_100)
weak_100 = remove_name_2(weak_100)
print(len(strong_742),len(strong_742[0]))
print(len(weak_742),len(weak_742[0]))
print(len(strong_100),len(strong_100[0]))
print(len(weak_100),len(weak_100[0]))
train_x_1484 = np.concatenate([strong_742, weak_742], axis=0)
test_x_200 = np.concatenate([strong_100, weak_100], axis=0)
print(len(train_x_1484),len(test_x_200))

# %%
train_y = np.concatenate([np.ones((len(enhancer_cv),)), np.zeros((len(non_cv),))], axis=0)  #竖向拼接
test_y = np.concatenate([np.ones((len(enhancer_ind),)), np.zeros((len(non_ind),))], axis=0)
print(train_y.shape,test_y.shape)

# %%
train_y_1484 = np.concatenate([np.ones((len(strong_742),)), np.zeros((len(weak_742),))], axis=0)  #竖向拼接
test_y_200 = np.concatenate([np.ones((len(strong_100),)), np.zeros((len(weak_100),))], axis=0)
print(train_y_1484.shape,test_y_200.shape)

# %%
def encode_matrix(seq_matrix):
    """将字符编码为整数
    """
    ind_to_char = ['A','T','C','G','N']
    char_to_ind = {char: i for i, char in enumerate(ind_to_char)}
    return [[char_to_ind[i] for i in s] for s in seq_matrix]

def decode_tensor(tensor):
    """将整数张量解码为DNA字符字符串
    """
    ind_to_char = ['A', 'T', 'C', 'G', 'N']
    sequence = [[ind_to_char[i] for i in sublist] for sublist in tensor.tolist()]
    return [''.join(sublist) for sublist in sequence]

# %%
train_x = encode_matrix(train_x)
test_x = encode_matrix(test_x)
train_x = np.array(train_x)
test_x = np.array(test_x)
print(train_x.shape,test_x.shape)

# %%
train_x_1484 = encode_matrix(train_x_1484)
test_x_200 = encode_matrix(test_x_200)
train_x_1484 = np.array(train_x_1484)
test_x_200 = np.array(test_x_200)
print(train_x_1484.shape,test_x_200.shape)

# %%
#定义SN、SP、ACC、MCC
def sn_sp_acc_mcc(true_label,predict_label,pos_label=1):
    import math
    pos_num = np.sum(true_label==pos_label)
    print('pos_num=',pos_num)
    neg_num = true_label.shape[0]-pos_num
    print('neg_num=',neg_num)
    tp =np.sum((true_label==pos_label) & (predict_label==pos_label))
    print('tp=',tp)
    tn = np.sum(true_label==predict_label)-tp
    print('tn=',tn)
    sn = tp/pos_num
    sp = tn/neg_num
    acc = (tp+tn)/(pos_num+neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=',fn)
    print('fp=',fp)
    
    tp = np.array(tp,dtype=np.float64)
    tn = np.array(tn,dtype=np.float64)
    fp = np.array(fp,dtype=np.float64)
    fn = np.array(fn,dtype=np.float64)
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn)))
    return sn,sp,acc,mcc

# %%
import torch
import torch.nn as nn

class Attention3D(nn.Module):
    def __init__(self, input_dim):
        super(Attention3D, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 计算注意力权重
        weights = self.W(x).squeeze(dim=-1)  # 将线性变换后的输出维度从 input_dim 转换为 1
        weights = torch.softmax(weights, dim=1)  # 使用 softmax 进行归一化，得到注意力权重

        # 加权求和
        weighted_sum = torch.sum(x * weights.unsqueeze(dim=-1), dim=1)  # 在第二维进行加权求和，保持维度一致

        return weighted_sum

# 示例用法


import torch
import torch.nn as nn

def resnet_identity_block(input_data, filters, kernel_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CNN层
    # print("input"+str(input_data.shape))
    x = nn.Conv1d(filters, filters, kernel_size, stride=1, padding=(kernel_size // 2)).to(device)(input_data)
    # print("x"+str(x.shape))
    x = nn.BatchNorm1d(filters).to(device)(x)  #批次标准化
    x = nn.ReLU().to(device)(x)
    # 第二层没有激活函数
    x = nn.Conv1d(filters, filters, kernel_size, stride=1, padding=(kernel_size // 2)-1).to(device)(x)
    # x = nn.BatchNorm1d(filters)(x)
    # print("x"+str(x.shape))
    # 两个张量相加
    x = x.to(device) + input_data.to(device)
    # 对相加的结果使用ReLU激活
    x = nn.ReLU().to(device)(x)
    # 返回结果
    return x

import torch
import torch.nn as nn

def resnet_convolutional_block(input_data, filters, kernel_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CNN层
    x = nn.Conv1d(input_data.size(1), filters, kernel_size, stride=2, padding=0).to(device)(input_data)
    x = nn.BatchNorm1d(filters).to(device)(x)  #批次标准化
    x = nn.ReLU().to(device)(x)
    # 第二层没有激活函数
    x = nn.Conv1d(filters, filters, kernel_size, padding=4).to(device)(x)
    x = nn.BatchNorm1d(filters).to(device)(x)
    # print(input_data.shape)
    X = nn.Conv1d(input_data.size(1), filters, kernel_size, stride=2, padding=1).to(device)(input_data)
    # print(X.shape)
    # 两个张量相加
    x = x.to(device) + X.to(device)
    # 对相加的结果使用ReLU激活
    x = nn.ReLU().to(device)(x)
    # 返回结果
    return x

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, DNATokenizer

class ResNetModel(nn.Module):
    def __init__(self, maxlen, max_features, embedding_dims, class_num):
        super(ResNetModel, self).__init__()
        self.conv = nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4)
        self.conv1 = nn.Conv1d(768, 384, kernel_size=3, stride=1, padding=1)
        self.embedding=nn.Embedding(200,32)
        self.bn = nn.BatchNorm1d(32)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.global_max_pool2 = nn.AdaptiveMaxPool1d(256)
        self.lstm = nn.LSTM(200, 32, bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(128, 32, bidirectional=True, batch_first=True)
        self.attention = Attention3D(768)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(1024+768, 128)
        self.fc5 = nn.Linear(128, 16)
        self.fc6 = nn.Linear(74880, 768)
        self.output = nn.Linear(16, class_num)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device    
        self.embedding = self.embedding.to(self.device)
        self.conv = self.conv.to(self.device)
        self.bn = self.bn.to(self.device)
        self.lstm = self.lstm.to(self.device)
        # self.lstm2 = self.lstm2.to(self.device)
        self.attention = self.attention.to(self.device)
        self.dropout = self.dropout.to(self.device)
        self.fc1 = self.fc1.to(self.device)
        self.fc2 = self.fc2.to(self.device)
        self.fc3 = self.fc3.to(self.device)
        self.flatten = nn.Flatten()

        self.output = self.output.to(self.device)
        self.dir_to_pretrained_model = "finetune_1123"
        self.config = BertConfig.from_pretrained('src/transformers/dnabert-config/bert-config-6/config.json')
        self.tokenizer = DNATokenizer.from_pretrained('dna6')
        self.model = BertModel.from_pretrained(self.dir_to_pretrained_model, config=self.config)
        self.cnn3_1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.cnn3_3 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=2)
        self.cnn3_5 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=7, padding=3)
    def forward(self, x1):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = decode_tensor(x1)
        results = []
        # DNABERT预训练模型得到输出
        for string in x:
            k_mers = [string[i:i+6] for i in range(len(string)-6+1)]
            processed_sequence = ' '.join(k_mers)
            model_input = self.tokenizer.encode_plus(processed_sequence, add_special_tokens=False, max_length=512)["input_ids"]
            model_input = torch.tensor(model_input, dtype=torch.long)
            model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
            model_input = model_input.to(device)
            output = self.model(model_input)
            results.append(output[0])
        x1=x1.to(device)
        #embedding部分网络架构
        embedding = self.embedding(x1)
        embedding1 = embedding.permute(0, 2, 1)  #BSD变为BDS
        conv_out = self.conv(embedding1)
        conv_out = self.bn(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = F.max_pool1d(conv_out, kernel_size=2, stride=1)
        resnet_out = self.lstm(conv_out)[0]
        resnet_out = self.dropout(resnet_out)
        resnet_out=self.flatten(resnet_out)
        resnet_out=self.fc2(resnet_out)
        #DNABERT预训练输出后的网络架构
        output_tensor = torch.cat(results, dim=0)
        output_tensor = output_tensor.permute(0, 2, 1)  #BSD变为BDS
        # output_tensor = self.conv1(output_tensor) #BSD变为BDS
        
        output_tensor1=self.cnn3_1(output_tensor)
        output_tensor2=self.cnn3_3(output_tensor)
        output_tensor3=self.cnn3_5(output_tensor)
        output_tensor1 = output_tensor1.permute(0, 2, 1)  
        output_tensor2 = output_tensor2.permute(0, 2, 1)  
        output_tensor3 = output_tensor3.permute(0, 2, 1)  
        output_tensor1 = F.relu(output_tensor1)
        output_tensor2 = F.relu(output_tensor2)
        output_tensor3 = F.relu(output_tensor3)

        output_tensor=torch.cat((output_tensor1,output_tensor2,output_tensor3),dim=1)
        output_tensor=self.flatten(output_tensor)
        output_tensor=self.fc6(output_tensor)
        output_tensor = self.dropout(output_tensor)
        #特征融合后输出
        fusion_out = torch.cat((output_tensor, resnet_out), dim=1)
        fusion_out = self.fc4(fusion_out)
        fusion_out= self.dropout(fusion_out)
        fusion_out = self.fc5(fusion_out)
        output = self.output(fusion_out)
        output=(output.reshape(output.size(0)))
        return output



def define_model():
    maxlen = 200
    max_features = 5
    embedding_dims = 32
    class_num = 1
    last_activation = nn.Sigmoid()
    model= ResNetModel(maxlen, max_features, embedding_dims, class_num)
    return model

# %%
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':#第一层
    model_layer1 = define_model()
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    # Define the model architecture
    model_layer1 = define_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_layer1.to(device)
    # Set the initial learning rate
    initial_learning_rate = 0.0001
    # Create DataLoader for training data
    train_dataset = TensorDataset(torch.Tensor(train_x).long(), torch.Tensor(train_y).long())
    from torch.utils.data import random_split

    # 划分训练集和验证集
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # 定义损失函数和优化器
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_layer1.parameters(), lr=initial_learning_rate)
    # 训练循环
    best_loss = float('inf')  # 初始化最小损失值为正无穷大
    epochs=30
    for epoch in range(epochs):
        model_layer1.train()  # 设置模型为训练模式
        # if epoch % 2 == 0:
        #     model_layer1.fc6.requires_grad_(False)
        # else:
        #     model_layer1.fc6.requires_grad_(True)
        epoch_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        for inputs, labels in epoch_progress_bar:
            labels = labels.to(device)
            optimizer.zero_grad()  # 清除梯度
            outputs = model_layer1(inputs)  # 前向传播
            loss = loss_function(outputs, labels.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            epoch_progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条上的损失值
        epoch_progress_bar.close()  # 关闭epoch进度条
        model_name = f'_model_{epoch}.pth'
        torch.save(model_layer1, model_name)