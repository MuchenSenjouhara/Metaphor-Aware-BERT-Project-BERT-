#Attention_Info.py

import torch  # 导入PyTorch核心库，用于张量计算和模型推理
import matplotlib.pyplot as plt  # 导入matplotlib绘图库，用于绘制可视化图表
import seaborn as sns  # 导入seaborn库，用于绘制更美观的热力图
from transformers import BertTokenizer, BertForSequenceClassification  # 导入BERT分词器和序列分类模型
import os  # 导入os模块，用于文件路径拼接操作

# ======================
# 1. 设备设置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测可用计算设备，优先使用GPU(CUDA)，否则使用CPU
print(f"Using device: {device}")  # 打印当前使用的计算设备，方便查看设备配置

# ======================
# 2. 加载模型与tokenizer
# ======================
model_path = "models/vuam_model"  # 定义训练好的隐喻检测模型保存路径
tokenizer = BertTokenizer.from_pretrained(model_path)  # 从指定路径加载预训练BERT分词器
model = BertForSequenceClassification.from_pretrained(model_path)  # 从指定路径加载预训练BERT序列分类模型
model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))  # 加载模型权重文件，map_location自动适配计算设备
model.to(device)  # 将模型移动到指定的计算设备（GPU/CPU）
model.eval()  # 将模型切换为评估模式，关闭Dropout等训练专属层

# ======================
# 3. 提取并可视化注意力
# ======================
def visualize_attention(input_ids, attention_probs, tokenizer, head_idx=0):  # 定义注意力可视化函数：接收输入ID、注意力概率、分词器和注意力头索引
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # 将输入ID转换为对应的token词汇，取第一个样本的token列表

    # 获取指定头的注意力矩阵
    attention_matrix = attention_probs.squeeze()[head_idx].cpu().numpy()  # 挤压多余维度，取指定注意力头的矩阵，转移到CPU并转换为NumPy数组（便于绘图）

    plt.figure(figsize=(10, 8))  # 创建绘图画布，指定画布大小为10*8英寸
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu", square=True)  # 绘制注意力热力图，横轴/纵轴为token，使用YlGnBu配色，保持正方形布局
    plt.xlabel("Tokens")  # 设置横轴标签为"Tokens"
    plt.ylabel("Tokens")  # 设置纵轴标签为"Tokens"
    plt.title(f"Attention Heatmap - Head {head_idx}")  # 设置图表标题，显示当前可视化的注意力头索引
    plt.show()  # 显示绘制完成的注意力热力图

# 示例代码：提取并可视化BERT的注意力
def get_attention_for_sentence(sentence, head_idx=0):  # 定义句子注意力提取函数：接收输入句子和注意力头索引
    inputs = tokenizer(  # 使用分词器对输入句子进行编码处理
        sentence,  # 传入需要可视化注意力的文本句子
        return_tensors="pt",  # 返回PyTorch张量格式的数据
        truncation=True,  # 开启截断功能，超过最大长度的文本将被截断
        max_length=128  # 指定最大序列长度为128
    ).to(device)  # 将编码后的张量移动到指定计算设备

    with torch.no_grad():  # 禁用梯度计算，节省内存并加快推理速度（评估阶段无需更新参数）
        outputs = model(**inputs, output_attentions=True)  # 模型前向传播，output_attentions=True开启注意力权重输出

    # 获取最后一层的注意力
    attention_probs = outputs.attentions[-1]  # 取BERT最后一层的注意力权重（通常最后一层注意力更具语义代表性）
    visualize_attention(inputs["input_ids"], attention_probs, tokenizer, head_idx)  # 调用可视化函数，绘制注意力热力图

# 运行示例
sentence = "She has a heart of stone."  # 定义示例句子（包含隐喻表达："heart of stone" 铁石心肠）
get_attention_for_sentence(sentence, head_idx=0)  # 调用函数，可视化该句子在第0个注意力头的注意力分布