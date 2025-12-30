#Data_Preprocessing.py

import json  # 导入json模块，用于数据的序列化与反序列化
from sklearn.model_selection import train_test_split  # 从sklearn导入数据集划分函数，用于拆分训练集和验证集
from datasets import load_dataset  # 从datasets库导入加载HF数据集的函数
import os  # 导入os模块，用于文件和目录的操作

# 加载 VUA 数据集（HF）
dataset = load_dataset("matejklemen/vuamc")  # 加载Hugging Face上的matejklemen/vuamc（VUA）隐喻数据集

def convert_to_sentence_labels(split):  # 定义函数：将数据集拆分（split）转换为带句子和隐喻标签的格式
    sentences = []  # 初始化空列表，用于存储转换后的句子及标签数据
    for sample in split:  # 遍历当前数据集拆分中的每一个样本
        words = sample["words"]  # 提取样本中的单词列表
        met_type = sample["met_type"]  # 提取样本中的隐喻类型标注信息

        sentence_label = 1 if any(  # 判定句子是否包含隐喻，是则标签为1，否则为0
            met["type"] == "mrw/met" for met in met_type  # 检查是否存在类型为"mrw/met"的隐喻标注
        ) else 0

        sentence = " ".join(words)  # 将单词列表拼接为完整的英文句子

        sentences.append({  # 将句子和对应的隐喻标签以字典形式添加到列表中
            "sentence": sentence,  # 存储拼接后的完整句子
            "metaphor_label": sentence_label  # 存储该句子的隐喻标签（0/1）
        })
    return sentences  # 返回转换后的所有句子及标签数据


# 使用 train 数据
all_data = convert_to_sentence_labels(dataset["train"])  # 对数据集的训练集拆分执行格式转换，得到全部训练数据

train_data, val_data = train_test_split(  # 将全部训练数据拆分为训练集和验证集
    all_data, test_size=0.1, random_state=42  # 验证集占比10%，随机种子42保证结果可复现
)

test_data = train_data[-2000:]  # 从训练集中截取最后2000条数据作为测试集

os.makedirs("data/processed", exist_ok=True)  # 创建数据保存目录，exist_ok=True避免目录已存在时报错

def save_jsonl(path, data):  # 定义函数：将数据以jsonl格式保存到指定路径
    with open(path, "w", encoding="utf-8") as f:  # 以写入模式打开文件，指定编码为utf-8
        for item in data:  # 遍历数据中的每一个条目
            f.write(json.dumps(item, ensure_ascii=False) + "\n")  # 将条目序列化为json字符串并换行写入，保证非ASCII字符正常显示

save_jsonl("data/processed/train.jsonl", train_data)  # 保存训练集数据到指定jsonl文件
save_jsonl("data/processed/validation.jsonl", val_data)  # 保存验证集数据到指定jsonl文件
save_jsonl("data/processed/test.jsonl", test_data)  # 保存测试集数据到指定jsonl文件

print("Data preprocessing complete (jsonl).")  # 打印提示信息，标识数据预处理完成