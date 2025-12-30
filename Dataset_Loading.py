#Dataset_Loading.py

import json  # 导入json模块，用于解析jsonl格式文件中的单行json数据
import torch  # 导入PyTorch核心库，用于创建张量和支持深度学习相关操作
from torch.utils.data import Dataset  # 从PyTorch数据模块导入Dataset基类，用于自定义数据集
from transformers import BertTokenizer  # 从transformers库导入BERT分词器，用于文本编码处理

class VUAMetaphorDataset(Dataset):  # 定义自定义数据集类，继承自PyTorch的Dataset基类，用于加载隐喻任务数据
    def __init__(self, file_path, tokenizer, max_length=128):  # 构造函数：初始化数据集的核心参数
        self.data = []  # 初始化空列表，用于存储从jsonl文件加载的所有数据条目
        with open(file_path) as f:  # 以默认只读模式打开指定路径的jsonl文件
            for line in f:  # 逐行遍历文件内容（jsonl格式每行对应一个独立json对象）
                self.data.append(json.loads(line))  # 解析单行json字符串并添加到数据列表中
        self.tokenizer = tokenizer  # 保存传入的BERT分词器实例，供后续文本编码使用
        self.max_length = max_length  # 保存最大序列长度，用于文本的截断和填充操作

    def __len__(self):  # 重写Dataset基类的__len__方法：返回数据集的总样本数量
        return len(self.data)  # 返回数据列表的长度，即数据集包含的样本总数

    def __getitem__(self, idx):  # 重写Dataset基类的__getitem__方法：根据索引idx获取单个样本
        item = self.data[idx]  # 根据索引idx提取对应的数据条目（字典格式）
        encodings = self.tokenizer(  # 使用BERT分词器对句子进行编码转换
            item["sentence"],  # 传入数据条目中的文本句子
            truncation=True,  # 开启截断功能，超过max_length的文本将被截断
            padding="max_length",  # 按max_length进行填充，不足长度的文本补PAD token
            max_length=self.max_length,  # 指定文本编码后的最大序列长度
            return_tensors="pt"  # 指定返回PyTorch张量（Tensor）格式的数据
        )
        label = item["metaphor_label"]  # 从数据条目中提取隐喻标签（0/1）
        encodings["labels"] = torch.tensor(label)  # 将标签转换为PyTorch张量，并添加到编码结果字典中
        return encodings  # 返回包含编码信息和标签的字典，作为单个样本数据
print("数据准备完成")