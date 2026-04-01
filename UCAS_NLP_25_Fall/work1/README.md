# Work1: 猫狗图像分类

## 训练方式

**答案：直接运行一条命令即可，无需分次训练。**

本项目会自动依次训练 CNN、RNN、DNN 三个模型。

## 快速开始

```bash
cd code
python train.py
```

## 训练过程

运行 `train.py` 后，程序会：

1. **自动依次训练**三个模型（CNN、RNN、DNN）
2. 每个模型训练 **10 个 epoch**
3. 在每个 epoch 后评估验证集性能
4. 训练完成后，结果自动保存到 `training_results.json`

## 输出结果

- **training_results.json**: 包含所有模型的训练结果
- **终端输出**: 显示每个模型的训练进度和验证指标

## 实验报告

训练完成后，可查看 `code/report.md` 了解：
- 各模型性能对比
- 最佳模型选择
- 改进建议
