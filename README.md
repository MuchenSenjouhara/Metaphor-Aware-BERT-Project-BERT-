# Metaphor-Aware BERT 隐喻识别项目

---

## 项目简介 / Project Overview

该项目基于 BERT 模型实现隐喻检测（Metaphor Detection），支持单句预测和批量文件预测，并提供 Attention 可视化功能，帮助理解模型的决策依据。  
This project implements metaphor detection using a BERT-based model. It supports single-sentence prediction and batch file prediction, with attention visualization to help interpret the model’s decisions.

---

## 功能 / Features

- **单句预测 / Single Sentence Prediction**：输入一句话即可预测其是否为隐喻，并展示高亮词汇。
- **批量预测 / Batch Prediction**：上传文本文件（每行一句），模型批量预测，并输出结果及高亮词。
- **Attention 可视化 / Attention Visualization**：可视化 BERT 模型的注意力权重，分析模型关注的词汇。
- **模型评估 / Model Evaluation**：提供准确率、精确率、召回率、F1-score 和混淆矩阵的评估功能。

---

## 安装依赖 / Installation

建议使用 Python 3.10+ 并创建虚拟环境：

```bash
conda create -n metaphor-env python=3.10 -y
conda activate metaphor-env
pip install -r requirements.txt
