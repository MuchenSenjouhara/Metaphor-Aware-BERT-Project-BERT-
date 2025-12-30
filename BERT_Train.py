#BERT_Train.py

import json  # 导入json模块，用于解析jsonl格式的数据
import torch  # 导入PyTorch核心库，用于张量计算和模型构建
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器核心类
from transformers import BertTokenizer, BertForSequenceClassification  # 导入BERT分词器和序列分类模型
from torch.optim import AdamW  # 导入AdamW优化器，常用于Transformer模型训练
from tqdm import tqdm  # 导入tqdm库，用于显示训练/验证进度条
import os  # 导入os模块，用于目录创建和路径操作

# ======================
# 1. Dataset 定义（只读 json）
# ======================
class VUAMetaphorDataset(Dataset):  # 定义自定义数据集类，继承自PyTorch的Dataset基类
    def __init__(self, file_path, tokenizer, max_length=128):  # 构造函数：初始化数据集参数
        self.data = []  # 初始化空列表，用于存储加载的jsonl数据

        with open(file_path, "r", encoding="utf-8") as f:  # 以只读模式打开指定文件，编码为utf-8
            for line in f:  # 逐行遍历文件内容（jsonl格式每行一个json对象）
                self.data.append(json.loads(line))  # 解析每行json字符串并添加到数据列表

        self.tokenizer = tokenizer  # 保存传入的BERT分词器
        self.max_length = max_length  # 保存最大序列长度，用于文本截断和填充

    def __len__(self):  # 重写Dataset基类方法：返回数据集样本总数
        return len(self.data)  # 返回数据列表的长度，即样本数量

    def __getitem__(self, idx):  # 重写Dataset基类方法：根据索引获取单个样本
        item = self.data[idx]  # 根据索引idx获取对应的数据条目

        encoding = self.tokenizer(  # 使用分词器对句子进行编码处理
            item["sentence"],  # 传入需要编码的文本句子
            truncation=True,  # 开启截断功能，超过max_length的文本将被截断
            padding="max_length",  # 按max_length进行填充，不足部分补PAD token
            max_length=self.max_length,  # 指定最大序列长度
            return_tensors="pt"  # 返回PyTorch张量格式的数据
        )

        return {  # 返回编码后的模型输入张量和标签张量
            "input_ids": encoding["input_ids"].squeeze(0),  # 词嵌入ID张量，挤压维度（去除多余的batch维度）
            "attention_mask": encoding["attention_mask"].squeeze(0),  # 注意力掩码张量，挤压多余维度
            "labels": torch.tensor(item["metaphor_label"], dtype=torch.long)  # 隐喻标签张量，指定为长整型
        }

# ======================
# 2. 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有可用GPU，优先使用CUDA，否则使用CPU
print(f"Using device: {device}")  # 打印当前使用的计算设备

# ======================
# 3. 模型 & tokenizer
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 加载预训练的bert-base-uncased分词器
model = BertForSequenceClassification.from_pretrained(  # 加载预训练的bert-base-uncased序列分类模型
    "bert-base-uncased",  # 指定预训练模型名称
    num_labels=2  # 指定分类任务的标签数量（隐喻/非隐喻二分类）
)
model.to(device)  # 将模型移动到指定的计算设备（GPU/CPU）

# ======================
# 4. DataLoader
# ======================
train_dataset = VUAMetaphorDataset(  # 实例化训练集数据集
    "data/processed/train.jsonl",  # 训练集jsonl文件路径
    tokenizer  # 传入已加载的BERT分词器
)
val_dataset = VUAMetaphorDataset(  # 实例化验证集数据集
    "data/processed/validation.jsonl",  # 验证集jsonl文件路径
    tokenizer  # 传入已加载的BERT分词器
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练集数据加载器，批次大小32，开启数据打乱
val_loader = DataLoader(val_dataset, batch_size=32)  # 创建验证集数据加载器，批次大小32，不打乱数据

# ======================
# 5. Optimizer
# ======================
optimizer = AdamW(model.parameters(), lr=2e-5)  # 初始化AdamW优化器，传入模型参数，学习率设置为2e-5（BERT常用学习率）

# ======================
# 6. 训练函数
# ======================
def train_epoch(model, loader, optimizer):  # 定义单轮训练函数：接收模型、数据加载器、优化器
    model.train()  # 将模型切换为训练模式（启用Dropout、BatchNorm等训练专属层）
    total_loss = 0  # 初始化总损失值，用于计算本轮平均损失

    progress = tqdm(loader, desc="Training", leave=False)  # 创建训练进度条，描述为"Training"，不保留进度条痕迹
    for batch in progress:  # 遍历数据加载器中的每个批次
        optimizer.zero_grad()  # 清空优化器的梯度缓存（避免上一批次梯度累积）

        batch = {k: v.to(device) for k, v in batch.items()}  # 将批次中所有张量移动到指定计算设备
        outputs = model(**batch)  # 模型前向传播，传入批次数据（自动匹配input_ids、attention_mask、labels）

        loss = outputs.loss  # 从模型输出中获取本轮批次的损失值
        loss.backward()  # 损失反向传播，计算模型参数的梯度
        optimizer.step()  # 优化器更新模型参数

        total_loss += loss.item()  # 累加批次损失值（转换为标量）
        progress.set_postfix(loss=loss.item())  # 在进度条上实时显示当前批次的损失值

    return total_loss / len(loader)  # 返回本轮训练的平均损失（总损失/批次数量）

# ======================
# 7. 验证函数
# ======================
def evaluate(model, loader):  # 定义验证函数：接收模型和验证集数据加载器
    model.eval()  # 将模型切换为评估模式（关闭Dropout、BatchNorm等训练专属层）
    correct, total = 0, 0  # 初始化正确预测数和总样本数，用于计算准确率

    with torch.no_grad():  # 禁用梯度计算，节省内存并加快计算速度（评估阶段无需更新参数）
        progress = tqdm(loader, desc="Validating", leave=False)  # 创建验证进度条，描述为"Validating"
        for batch in progress:  # 遍历验证集数据加载器中的每个批次
            batch = {k: v.to(device) for k, v in batch.items()}  # 将批次中所有张量移动到指定计算设备
            outputs = model(**batch)  # 模型前向传播，获取预测结果

            preds = torch.argmax(outputs.logits, dim=1)  # 对模型输出的logits取argmax，得到预测标签（dim=1为标签维度）
            correct += (preds == batch["labels"]).sum().item()  # 累加预测正确的样本数量
            total += batch["labels"].size(0)  # 累加当前批次的总样本数量

    return correct / total  # 返回验证集准确率（正确数/总数）

# ======================
# 8. 主训练循环
# ======================
EPOCHS = 3  # 定义总训练轮数为3轮

for epoch in range(EPOCHS):  # 遍历每一轮训练
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")  # 打印当前训练轮数（直观显示，从1开始计数）
    train_loss = train_epoch(model, train_loader, optimizer)  # 执行单轮训练，获取训练平均损失
    val_acc = evaluate(model, val_loader)  # 执行验证，获取验证集准确率

    print(f"Train loss: {train_loss:.4f}")  # 打印本轮训练平均损失，保留4位小数
    print(f"Val accuracy: {val_acc:.4f}")  # 打印本轮验证集准确率，保留4位小数

# ======================
# 9. 保存模型
# ======================
os.makedirs("models/vuam_model", exist_ok=True)  # 创建模型保存目录，exist_ok=True避免目录已存在时报错

torch.save(model.state_dict(), "models/vuam_model/model.pth")  # 保存模型的状态字典（参数）到指定路径
model.save_pretrained("models/vuam_model")  # 将模型以Hugging Face格式保存到指定目录
tokenizer.save_pretrained("models/vuam_model")  # 将分词器以Hugging Face格式保存到指定目录

print("Model saved to models/vuam_model")  # 打印提示信息，标识模型保存完成