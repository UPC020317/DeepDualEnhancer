if __name__ == '__main__':#第一层
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from ReadData import get_train_data, encode_matrix ,decode_tensor,get_train_genomic_data
    from model_genomic import define_model



    # 定义要读取的六个 CSV 文件列表
    csv_files = ['./processed_data_nosame_finally_genomic_model/train_all.csv']
    # 调用函数进行读取和合并
    genomic_feature,seqs,enh_idx,label = get_train_genomic_data(csv_files)
    train_dataset = TensorDataset(torch.Tensor(genomic_feature).long(),torch.Tensor(seqs).long(),torch.Tensor(enh_idx).long(), torch.Tensor(label).long())
    from torch.utils.data import random_split
    # 划分训练集和验证集
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)







    model_layer1 = define_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_layer1.to(device)
    # Set the initial learning rate
    initial_learning_rate = 0.0001
    # 定义损失函数和优化器
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_layer1.parameters(), lr=initial_learning_rate)
    # 训练循环
    epochs=30





    for epoch in range(epochs):
        epoch_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        for genomic_feature, seqs, enh_idx,labels in epoch_progress_bar:
            labels = labels.to(device)
            optimizer.zero_grad()  # 清除梯度
            outputs = model_layer1(genomic_feature, seqs,enh_idx)  # 前向传播
            loss = loss_function(outputs, labels.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            epoch_progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条上的损失值
        epoch_progress_bar.close()  # 关闭epoch进度条
        model_name = f'_model_genomic_{epoch}.pth'
        torch.save(model_layer1, model_name)