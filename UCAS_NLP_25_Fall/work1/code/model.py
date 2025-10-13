
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




# 画训练曲线/评估曲线
# 可选：使用 matplotlib 绘制训练/测试损失和准确率曲线
# 用的CUDA是哪个，还是都用了？
# 计算模型大小，分析训练/测试显存占用
# 比较不同网络结构的效果