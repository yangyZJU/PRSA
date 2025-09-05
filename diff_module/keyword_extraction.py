import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

# 如果还没有下载停用词列表，需要先下载
nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords(text):
    # 分词
    words = word_tokenize(text)

    # 移除停用词和标点符号
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # 计算词频
    freq_dist = FreqDist(filtered_words)

    # 返回最常见的词汇
    return [word for word, freq in freq_dist.most_common(10)]

# 示例文本
text = "The prompt is likely asking for the creation of a marketing email or promotional message targeting individuals interested in fitness (the customer persona) to promote fitness equipment. The content of the prompt may involve specifying the product or service (fitness equipment), identifying the target audience (fitness enthusiasts), and then instructing the AI to draft a persuasive email that highlights the benefits of the fitness equipment, offers exclusive deals or promotions, and encourages the recipient to take action, such as visiting a website or making a purchase. Essentially, it's a request to generate a marketing email tailored to a specific audience and product/service."

# 提取关键词
keywords = extract_keywords(text)

print("Keywords:", keywords)
