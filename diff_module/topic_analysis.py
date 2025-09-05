import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import gensim
import re

import spacy
from collections import Counter
# 下载NLTK数据
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # 分词
    words = word_tokenize(text)

    # 移除停用词
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    
    return filtered_words



def topic_identification(documents):
    # 加载Spacy的英文模型
    nlp = spacy.load("en_core_web_sm")

    # 初始化一个计数器来记录关键词
    keywords = Counter()

    for document in documents:
        # 使用Spacy进行文档处理
        doc = nlp(document)

        # 提取名词和专有名词
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                keywords[token.lemma_] += 1

        # 可以根据需要提取命名实体（如组织、人物）
        for ent in doc.ents:
            keywords[ent.text] += 1

    # 返回出现频次最高的前几个关键词
    return [keyword for keyword, _ in keywords.most_common(5)]
'''
def topic_identification(documents, num_words=4):
    # 预处理文档
    texts = [preprocess_text(document) for document in documents]

    # 创建字典
    dictionary = corpora.Dictionary(texts)

    # 创建语料库
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 应用LDA模型
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)

    # 显示主题
    topics = lda_model.print_topics(num_words)

    extracted_data = re.findall(r"(\d\.\d+)\*(\w+)", str(topics[0][1]).replace("\"", ""))
    data_dict = {word: float(score) for score, word in extracted_data}

    return data_dict
'''

if __name__ == '__main__':
    documents = [f"""
    Product/Service: fitness equipment.
    Customer Persona: fitness enthusiasts. 
    """]
    print(topic_identification(documents, num_words=1))
