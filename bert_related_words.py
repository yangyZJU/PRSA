


from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def find_similar_words(theme_word, given_text, top_n=5):
    theme_embedding = embed_text(theme_word)

    tokens = tokenizer.tokenize(given_text)
    unique_tokens = list(set(tokens))

    token_embeddings = [embed_text(token) for token in unique_tokens]
    similarities = [cosine_similarity(theme_embedding, token_embedding)[0][0] for token_embedding in token_embeddings]

    most_similar_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n] 
    most_similar_words = [unique_tokens[i] for i in most_similar_indices]

    return most_similar_words

def batch_find_sim_words(word_list, given_text):
    sim_list = []
    for word in word_list:
        sim_list += find_similar_words(word, given_text)
    return sim_list

if __name__ == "__main__":
    #theme_word = "Online cooking courses"
    theme_word = ['Foodie Media', 'Online cooking courses', 'Beginner home cooks', 'Generate course signups']
    given_text = """Please create a series of promotional content for the Wolf Pack Coaching Program, a 12-week online sales training course targeted at aspiring entrepreneurs and sales professionals. The content should include persuasive and urgent emails, a compelling landing page text, engaging social media posts, descriptive product descriptions, and catchy promotional materials. The tone should create a sense of scarcity and exclusivity, emphasizing the urgency to join the program."""

    
    most_similar_words = batch_find_sim_words(theme_word, given_text)
    print(most_similar_words)