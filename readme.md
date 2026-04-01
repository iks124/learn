# Learn Repository

本仓库包含中科院自然语言处理课程作业和学习项目。

## 课程作业 (UCAS_NLP_25_Fall)

### Work1: 猫狗图像分类

**训练说明**：该项目使用**一条命令**即可完成所有模型的训练，无需分次训练。

训练脚本会**自动依次训练** CNN、RNN、DNN 三个模型，并将结果保存到 `training_results.json` 文件中。

**训练步骤**：

```bash
cd UCAS_NLP_25_Fall/work1/code
python train.py
```

训练过程：
1. 自动依次训练 CNN、RNN、DNN 三个模型
2. 每个模型训练 10 个 epoch
3. 训练结果自动保存到 `training_results.json`
4. 查看 `report.md` 了解实验结果分析

### Work2: CBOW 词向量训练

**训练说明**：该项目使用**一条命令**即可完成训练，无需分次训练。

**训练步骤**：

```bash
cd UCAS_NLP_25_Fall/work2/code
python cbow_ns.py
```

训练过程：
1. 读取 `../data/zh.txt` 语料文件
2. 训练 CBOW 模型（带负采样）
3. 训练 10 个 epoch
4. 词向量自动保存到 `zh_vectors.txt`

---

## Git 基本操作

### 1. Initialize Git Repository

```bash
git init
```

### 2. Add Files

```bash
git add .
```

### 3. Commit Changes

```bash
git commit -m "Initial commit"
```

### 4. Add Remote Repository

```bash
git remote add origin <your-repo-url>
```

### 5. Push to Remote

```bash
git push -u origin master
```

> Replace `<your-repo-url>` with your actual repository URL.

### config git 

```bash
git config --global user.name "iks"
git config --global user.email "ucasrhk@gmail.com"
```