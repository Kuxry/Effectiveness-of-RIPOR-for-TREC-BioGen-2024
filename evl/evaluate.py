
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# 定义嵌入生成函数，将文本转换为BERT嵌入向量
def get_embedding(text):
    # 对文本进行编码并获取词汇 ID
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # 使用模型生成嵌入
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取句子的嵌入（使用最后一个隐藏层的均值）
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# 计算标题和摘要的余弦相似度
def calculate_similarity(title, abstract):
    # 获取标题和摘要的嵌入
    title_embedding = get_embedding(title)
    abstract_embedding = get_embedding(abstract)

    # 计算余弦相似度
    similarity = cosine_similarity(title_embedding, abstract_embedding)
    return similarity[0][0]


# 示例：计算单个标题和摘要之间的相似度
title = "Example title"
abstract = "Example abstract relevant to the title"
similarity_score = calculate_similarity(title, abstract)
print(f"Semantic Similarity Score: {similarity_score}")

# 评估整个数据集的平均相似度分数
# 假设 dataset 是一个包含 (title, abstract) 元组的列表
dataset = [
    ("Title 1", "Abstract 1"),
    ("Title 2", "Abstract 2"),
    # 添加更多数据...
]

# 计算所有数据点的相似度分数
similarity_scores = [calculate_similarity(title, abstract) for title, abstract in dataset]
average_similarity = sum(similarity_scores) / len(similarity_scores)
print(f"Average Semantic Similarity Score: {average_similarity}")
