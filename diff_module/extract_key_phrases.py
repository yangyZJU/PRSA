import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

# 如果还没有下载所需数据包，需要先下载
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_key_phrases(text):
    # 分词
    words = word_tokenize(text)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # 词性标注
    tagged = pos_tag(filtered_words)

    # 定义短语的正则表达式
    pattern = "NP: {<DT>?<JJ>*<NN>}"

    # 分析结构
    cp = RegexpParser(pattern)
    cs = cp.parse(tagged)

    # 提取短语
    key_phrases = []
    for subtree in cs.subtrees():
        if subtree.label() == 'NP':
            key_phrases.append(' '.join(word for word, tag in subtree.leaves()))

    return key_phrases

# 示例文本
text = "The prompt is likely asking for the creation of a marketing email or promotional message targeting individuals interested in fitness (the customer persona) to promote fitness equipment. The content of the prompt may involve specifying the product or service (fitness equipment), identifying the target audience (fitness enthusiasts), and then instructing the AI to draft a persuasive email that highlights the benefits of the fitness equipment, offers exclusive deals or promotions, and encourages the recipient to take action, such as visiting a website or making a purchase. Essentially, it's a request to generate a marketing email tailored to a specific audience and product/service."

# 提取关键短语
key_phrases = extract_key_phrases(text)

print("Key Phrases:", key_phrases)
