#gradio_app.py

import os  # 导入os模块，用于文件路径拼接和系统路径操作
import json  # 导入json模块，支持json数据格式的解析与序列化（本代码中备用）
import torch  # 导入PyTorch核心库，用于张量计算和模型推理
import gradio as gr  # 导入Gradio库，用于快速构建可视化Web交互界面
from transformers import BertTokenizer, BertForSequenceClassification  # 导入BERT分词器和序列分类模型
from tqdm import tqdm  # 导入tqdm库，用于显示批量预测的进度条

# ======================
# 1. 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测可用计算设备，优先使用GPU(CUDA)，否则使用CPU
print(f"Using device: {device}")  # 打印当前使用的计算设备，方便查看设备配置

# ======================
# 2. 加载模型
# ======================
model_path = "models/vuam_model"  # 定义预训练模型的保存路径
tokenizer = BertTokenizer.from_pretrained(model_path)  # 从指定路径加载预训练的BERT分词器
model = BertForSequenceClassification.from_pretrained(model_path)  # 从指定路径加载预训练的BERT序列分类模型
model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))  # 加载模型权重文件，map_location自动适配计算设备
model.to(device)  # 将模型移动到指定的计算设备（GPU/CPU）
model.eval()  # 将模型切换为评估模式，关闭Dropout等训练专属层

# ======================
# 3. 单句预测函数 + 高亮 HTML
# ======================
def predict_with_highlight(sentence, top_k=3):  # 定义单句预测函数，接收输入句子和高亮词数量
    inputs = tokenizer(  # 使用分词器对输入句子进行编码处理
        sentence,  # 传入需要预测的文本句子
        return_tensors="pt",  # 返回PyTorch张量格式的数据
        truncation=True,  # 开启截断功能，超过最大长度的文本将被截断
        max_length=128  # 指定最大序列长度为128
    ).to(device)  # 将编码后的张量移动到指定计算设备

    with torch.no_grad():  # 禁用梯度计算，节省内存并加快推理速度（评估阶段无需更新参数）
        outputs = model(**inputs, output_attentions=True)  # 模型前向传播，output_attentions=True开启注意力权重输出

    probs = torch.softmax(outputs.logits, dim=1)[0]  # 对模型输出的logits做softmax转换为概率分布，取第一个样本（单句）
    metaphor_prob = probs[1].item()  # 获取隐喻标签（索引1）的概率值，并转换为Python标量

    # 使用最后一层 attention 作为 proxy
    attention = outputs.attentions[-1].mean(dim=1)[0]  # 取最后一层注意力权重，按注意力头维度平均，再取第一个样本，得到[seq, seq]形状的注意力矩阵
    token_scores = attention.mean(dim=0)  # 按序列维度平均，得到每个token的注意力得分

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # 将input_ids转换为对应的token词汇

    # top-k 高亮词索引
    topk_indices = torch.topk(token_scores, k=min(top_k, len(token_scores))).indices  # 获取注意力得分最高的top-k个token索引，防止k值超过token总数
    highlight_set = {tokens[i] for i in topk_indices if tokens[i] not in ["[CLS]", "[SEP]"]}  # 筛选出非特殊标记([CLS]/[SEP])的高亮token，存入集合

    # 构造高亮 HTML
    highlighted_sentence = ""  # 初始化高亮句子的HTML字符串
    for tok in tokens:  # 遍历所有编码后的token
        if tok in ["[CLS]", "[SEP]"]:  # 跳过特殊标记token，不显示在结果中
            continue
        display_tok = tok.replace("##", "")  # 去除BERT分词的子词标记（##），还原完整词汇
        if display_tok in highlight_set:  # 判断当前token是否在高亮集合中
            highlighted_sentence += f'<span style="color:red;font-weight:bold">{display_tok}</span> '  # 用红色粗体包裹高亮词汇
        else:
            highlighted_sentence += display_tok + " "  # 非高亮词汇直接拼接

    label = "Metaphor" if metaphor_prob > 0.5 else "Literal"  # 根据隐喻概率判断标签（大于0.5为隐喻，否则为字面义）
    return label, round(metaphor_prob, 4), highlighted_sentence.strip()  # 返回预测标签、保留4位小数的隐喻概率、去除末尾空格的高亮HTML句子

# ======================
# 4. 批量预测函数
# ======================
def batch_predict(file, top_k=3):  # 定义批量预测函数，接收上传文件和高亮词数量
    results = []  # 初始化空列表，用于存储批量预测结果

    # Gradio 文件上传返回的 file 是 NamedString 类型，我们直接从 file 中读取内容
    file_content = file.name  # 获取上传文件的本地路径字符串

    # 打开并读取文件内容
    with open(file_content, "r", encoding="utf-8") as f:  # 以只读模式打开文件，指定编码为utf-8
        lines = f.read().splitlines()  # 读取文件所有内容并按行分割，去除每行末尾的换行符

    for line in tqdm(lines, desc="Predicting"):  # 遍历每行文本，用tqdm显示预测进度条
        sentence = line.strip()  # 去除句子前后的空格、换行符等空白字符
        if not sentence:  # 跳过空行，避免无效预测
            continue
        label, prob, highlighted_html = predict_with_highlight(sentence, top_k)  # 调用单句预测函数，获取预测结果
        results.append({  # 将当前句子的预测结果以字典形式添加到结果列表
            "sentence": sentence,  # 存储原始输入句子
            "label": label,  # 存储预测标签
            "prob": prob,  # 存储隐喻概率
            "highlighted_html": highlighted_html  # 存储高亮HTML字符串
        })
    return results  # 返回批量预测的完整结果列表

# ======================
# 5. Gradio 界面
# ======================
with gr.Blocks() as demo:  # 创建Gradio Blocks布局（支持多标签、复杂组件排列）
    gr.Markdown("## Metaphor Detection with BERT + Attention Highlights")  # 添加Markdown标题，说明工具功能

    with gr.Tab("单句预测"):  # 创建“单句预测”标签页
        input_text = gr.Textbox(label="输入句子")  # 创建文本输入框，用于输入单个待预测句子
        top_k_slider = gr.Slider(minimum=1, maximum=10, step=1, label="高亮 top-k 词")  # 创建滑动条，用于选择高亮词数量
        output_label = gr.Textbox(label="预测标签")  # 创建文本输出框，用于显示预测标签（Metaphor/Literal）
        output_prob = gr.Textbox(label="隐喻概率")  # 创建文本输出框，用于显示隐喻概率值
        output_highlight = gr.HTML(label="高亮词汇")  # 创建HTML输出组件，用于显示带高亮样式的句子
        btn = gr.Button("预测")  # 创建预测按钮
        btn.click(  # 绑定按钮点击事件
            fn=predict_with_highlight,  # 指定点击后执行的函数（单句预测函数）
            inputs=[input_text, top_k_slider],  # 指定函数输入对应的Gradio组件
            outputs=[output_label, output_prob, output_highlight]  # 指定函数输出对应的Gradio组件
        )

    with gr.Tab("批量预测"):  # 创建“批量预测”标签页
        file_input = gr.File(label="上传文本文件，每行一句")  # 创建文件上传组件，用于上传批量待预测的文本文件
        batch_top_k = gr.Slider(minimum=1, maximum=10, step=1, label="批量预测 top-k 高亮")  # 创建滑动条，设置批量预测的高亮词数量
        batch_output = gr.HTML(label="批量预测结果")  # 创建HTML输出组件，用于显示批量预测的高亮结果

        def batch_display(file, top_k):  # 定义批量预测结果展示函数，处理文件并生成HTML展示内容
            results = batch_predict(file, top_k)  # 调用批量预测函数，获取预测结果列表
            html_lines = []  # 初始化空列表，用于存储每条结果的HTML字符串
            for r in results:  # 遍历批量预测结果
                html_lines.append(f'<p>{r["highlighted_html"]} <b>({r["label"]}, {r["prob"]})</b></p>')  # 拼接每条结果的HTML格式字符串
            return "".join(html_lines)  # 将所有结果的HTML字符串拼接后返回

        batch_btn = gr.Button("批量预测")  # 创建批量预测按钮
        batch_btn.click(batch_display, inputs=[file_input, batch_top_k], outputs=batch_output)  # 绑定按钮点击事件，执行批量预测并展示结果

demo.launch()  # 启动Gradio Web界面，默认在本地端口运行