import csv
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess


# 设置NLTK数据存储路径
nltk.data.path.append(r"C:\Users\wcx\AppData\Roaming\nltk_data")

def safe_nltk_download(pkg):
    try:
        nltk.data.find(f"{'corpora' if pkg in ['stopwords', 'wordnet'] else 'tokenizers'}/{pkg}")
        print(f"{pkg} 已存在，跳过下载。")
    except LookupError:
        nltk.download(pkg)
        print(f"{pkg} 下载完成。")

try:
    safe_nltk_download('punkt')
    safe_nltk_download('stopwords')
    safe_nltk_download('wordnet')
    print("NLTK数据包检测与下载完成！")
except Exception as e:
    print(f"下载NLTK数据包时出错: {e}")

def load_data(file_path, num_samples=10):
    """加载验证集数据,只保留英文文本,限制条数"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                content = row[1]
                # 判断文本是否为英文（英文字母比例大于60%）
                letters = re.findall(r'[A-Za-z]', content)
                ratio = len(letters) / max(len(content), 1)
                if ratio > 0.6:
                    texts.append(content)
                    if len(texts) >= num_samples:
                        break
    return texts

def preprocess_text(text):
    """文本预处理"""
    # 转小写并去除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # 按空格切分
    tokens = text.split()
    
    # 去停用词和短词
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return tokens

def analyze_topic_with_llm(topic_words):
    """使用大模型分析主题内容"""
    prompt = f"""作为一个主题分析专家，请分析以下关键词组成的主题代表什么内容：
关键词：{', '.join(topic_words)}
请简要概括主题内容（50字以内）。"""
    
    try:
        result = subprocess.run(
            ['ollama', 'run', 'gemma3', prompt],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.stdout.strip() if result.returncode == 0 else "无法解析主题"
    except Exception as e:
        print(f"调用模型出错: {e}")
        return "分析失败"

def main():
    # 1. 加载数据
    file_path = r"c:\Users\wcx\Desktop\期末大作业\验证集.csv"
    texts = load_data(file_path, num_samples=10)  # 限制10条
    print(f"加载数据完成，共{len(texts)}条文本")
    print("texts内容:", texts)
    
    # 2. 数据预处理
    processed_texts = [preprocess_text(text) for text in texts]
    
    # 3. 构建词典和语料库
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # 4. 训练LDA模型
    num_topics = 3  # 由于数据量少，减少主题数量
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15
    )
    
    # 5. 可视化分析
    # 5.1 pyLDAvis可视化
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    import webbrowser
    webbrowser.open('lda_visualization.html')
    
    # 5.2 生成词云
    plt.figure(figsize=(20, 4))
    for i in range(num_topics):
        plt.subplot(1, 5, i+1)
        topic_words = dict(lda_model.show_topic(i, 20))
        wordcloud = WordCloud(
            width=400, height=400,
            background_color='white'
        ).generate_from_frequencies(topic_words)
        plt.imshow(wordcloud)
        plt.title(f'Topic {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('topic_wordclouds.png')
    
    # 5.3 热力图
    doc_topics = np.zeros((len(texts), num_topics))
    for i, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow)
        for topic_id, prob in topic_dist:
            doc_topics[i, topic_id] = prob
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(doc_topics[:50], # 只显示前50篇文档
                cmap='YlOrRd',
                xticklabels=[f'Topic {i+1}' for i in range(num_topics)])
    plt.title('Document-Topic Distribution')
    plt.savefig('topic_distribution.png')
    
    # 6. 使用大模型分析主题
    print("\n=== 主题内容分析 ===")
    for i in range(num_topics):
        topic_words = [w for w, _ in lda_model.show_topic(i, 10)]
        topic_analysis = analyze_topic_with_llm(topic_words)
        print(f"\nTopic {i+1}:")
        print(f"关键词: {', '.join(topic_words)}")
        print(f"主题解释: {topic_analysis}")

if __name__ == "__main__":
    main()
