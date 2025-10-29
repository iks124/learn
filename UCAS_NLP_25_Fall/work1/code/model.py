
import torch.nn as nn

# -------------------------
# Model
# -------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 输入需为 224x224
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 更稳妥
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, hidden_size=64, output_size=2, num_layers=2):
        super(RNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 卷积层提取特征
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 池化层
        self.rnn = nn.RNN(112 * 112, hidden_size, num_layers, batch_first=True)  # RNN 层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状: (batch, 3, 224, 224)
        x = self.pool(self.relu(self.conv(x)))  # 卷积 + 池化，输出形状: (batch, 16, 112, 112)
        x = x.view(x.size(0), 16, -1)  # 展平为 RNN 的输入形状: (batch, seq_len=1, input_size)
        out, _ = self.rnn(x)  # RNN 层，输出形状: (batch, seq_len=1, hidden_size)
        out = self.relu(out[:, -1, :])  # 取最后一个时间步的输出，形状: (batch, hidden_size)
        out = self.fc(out)  # 全连接层，输出形状: (batch, output_size)
        return out
    
class DNN(nn.Module):
    def __init__(self, input_size=3 * 224 * 224, hidden_size=2048, output_size=2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 全连接层 1
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 全连接层 2
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)  # 全连接层 3
        self.fc4 = nn.Linear(hidden_size // 4, output_size)  # 全连接层 4
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 输入形状: (batch, 3, 224, 224)
        x = x.view(x.size(0), -1)  # 展平输入，形状: (batch, 3 * 224 * 224)
        x = self.relu(self.fc1(x))  # 全连接层 1，输出形状: (batch, hidden_size)
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc2(x))  # 全连接层 2，输出形状: (batch, hidden_size // 2)
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc3(x))  # 全连接层 3，输出形状: (batch, hidden_size // 4)
        x = self.fc4(x)  # 全连接层 4，输出形状: (batch, output_size)
        return x




# 画训练曲线/评估曲线
# 可选：使用 matplotlib 绘制训练/测试损失和准确率曲线
# 用的CUDA是哪个，还是都用了？
# 计算模型大小，分析训练/测试显存占用
# 比较不同网络结构的效果