# 导入所需的库
import sys
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 定义一个函数，读取文本文件并返回其内容
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# 定义一个函数，计算两个文本之间的余弦相似度
def text_similarity(text1, text2):
    # 创建一个TfidfVectorizer对象，用于把文本转换为向量
    vectorizer = TfidfVectorizer()
    # 把两个文本放在一个列表中，并转换为向量矩阵
    matrix = vectorizer.fit_transform([text1, text2])
    # 计算向量矩阵中第一行和第二行之间的余弦相似度，并返回结果
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])
    return similarity[0][0]

# 获取命令行参数，即两个文本文件的路径
file1 = sys.argv[1]
file2 = sys.argv[2]

# 读取两个文本文件的内容
text1 = read_file(file1)
text2 = read_file(file2)

# 计算两个文本之间的余弦相似度，并打印结果
similarity = text_similarity(text1, text2)
print(f"The cosine similarity between {file1} and {file2} is {similarity:.4f}")
