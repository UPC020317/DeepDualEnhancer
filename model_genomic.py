# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.lstm = nn.LSTM(180, 90, bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(128, 32, bidirectional=True, batch_first=True)
        self.attention = Attention3D(768)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128+768, 128)
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
        self.flatten = nn.Flatten()

        self.output = self.output.to(self.device)
        self.dir_to_pretrained_model = "new_dataset_pretrained_model"
        self.config = BertConfig.from_pretrained('src/transformers/dnabert-config/bert-config-6/config.json')
        self.tokenizer = DNATokenizer.from_pretrained('dna6')
        self.model = BertModel.from_pretrained(self.dir_to_pretrained_model, config=self.config)
        self.cnn3_1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.cnn3_3 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=2)
        self.cnn3_5 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=7, padding=3)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device  
        in_dim=8
        cnn_channels= [180]
        cnn_sizes=[11]
        cnn_pool= [10]
        enc_layers= 3
        num_heads= 6
        d_inner= 256
        da= 64
        r= 32
        att_C= 0.1
        fc= [128]
        fc_dropout= 0.2
        self.cnn = nn.ModuleList()
        self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_dim, 
                        out_channels=cnn_channels[0], 
                        kernel_size=cnn_sizes[0], 
                        padding=cnn_sizes[0] // 2),
                    nn.BatchNorm1d(cnn_channels[0]),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(cnn_pool[0])
                )
            )
        # seq_len //= cnn_pool[0]
        for i in range(len(cnn_sizes) - 1):
            self.cnn.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=cnn_channels[i], 
                            out_channels=cnn_channels[i + 1], 
                            kernel_size=cnn_sizes[i + 1],
                            padding=cnn_sizes[i + 1] // 2),
                        nn.BatchNorm1d(cnn_channels[i + 1]),
                        nn.LeakyReLU(),
                        nn.MaxPool1d(cnn_pool[i + 1])
                )
            )
        self.cnn = self.cnn.to(self.device)
            # seq_len //= cnn_pool[i + 1]
        enc_layer = nn.TransformerEncoderLayer(
                d_model=cnn_channels[-1],
                nhead=num_heads,
                dim_feedforward=d_inner,
                batch_first=True
            )

           
        self.encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=enc_layers
                )
        self.encoder = self.encoder.to(self.device)

        self.da = da
        self.r = r
        self.att_C = att_C
        self.att_first = nn.Linear(cnn_channels[-1], da)
        self.att_first.bias.data.fill_(0)
        self.att_second = nn.Linear(da, r)
        self.att_second.bias.data.fill_(0)
        self.att_first = self.att_first.to(self.device)
        self.att_second = self.att_second.to(self.device)

        # if fc[-1] != 1:
        #     fc.append(1)
        self.fc = nn.ModuleList()
        self.fc.append(
                nn.Sequential(
                    nn.Dropout(p=fc_dropout),
                    nn.Linear(cnn_channels[-1] * 3, fc[0])
                )
            )

        for i in range(len(fc) - 1):
            self.fc.append(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(fc[i], fc[i + 1])
                    )
                )
        self.fc.append(nn.Sigmoid())






















    def forward(self, feats,x2,enh_idx):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = decode_tensor(x2)
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


        #DNA序列处理框架部分
        #DNABERT预训练输出后的网络架构
        output_tensor = torch.cat(results, dim=0)
        output_tensor = output_tensor.permute(0, 2, 1)  #BSD变为BDS
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



        # enh_idx = torch.tensor(enh_idx, dtype=torch.float64)
        # feats = torch.tensor(feats, dtype=torch.float64)
        feats = feats.float()

        enh_idx = enh_idx.to(device)
        feats = feats.to(device)

        #基因组特征处理部分
        div = 1
        for cnn in  self.cnn:
            div *= cnn[-1].kernel_size
            enh_idx = torch.div(enh_idx, cnn[-1].kernel_size, rounding_mode="trunc")
            feats = cnn(feats)
        feats = feats.transpose(1, 2) # -> (B, S, D)
        batch_size, seq_len, feat_dim = feats.size()

        feats = self.encoder(feats) # (B, S, D)
        # feats = self.lstm(feats) # (B, S, D)
        # feats, _ = feats

        out = torch.tanh(self.att_first(feats)) # (B, S, da)
        out = F.softmax(self.att_second(out), 1) # (B, S, r)
        att = out.transpose(1, 2) # (B, r, S)
        del out
        seq_embed = torch.matmul(att, feats) # (B, r, D)
        base_idx = seq_len * torch.arange(batch_size) # .to(feats.device)
        base_idx = base_idx.to(device)
        enh_idx = enh_idx.long().view(batch_size) + base_idx
        feats = feats.reshape(-1, feat_dim)
        seq_embed = torch.cat((
            feats[enh_idx, :].view(batch_size, -1), 
            # feats[prom_idx, :].view(batch_size, -1),
            seq_embed.mean(dim=1).view(batch_size, -1),
            seq_embed.max(dim=1)[0].view(batch_size, -1)
        ), axis=1)
        del feats
        for fc in self.fc:
            seq_embed = fc(seq_embed)

        #特征融合后输出
        fusion_out = torch.cat((output_tensor, seq_embed), dim=1)
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
