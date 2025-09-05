from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


w2v_model = KeyedVectors.load_word2vec_format('tool/GoogleNews-vectors-negative300.bin.gz', binary=True)

def find_similar_words(theme_word, given_text, model, threshold=0.75):

    if theme_word not in model:
        raise ValueError(f"The word '{theme_word}' is not in the vocabulary.")
    theme_vector = model[theme_word].reshape(1, -1)
    tokens = given_text.split()
    similarities = {}
    for token in tokens:
        if token in model:
            token_vector = model[token].reshape(1, -1)
            similarity = cosine_similarity(theme_vector, token_vector)[0][0]
            if similarity > threshold:
                similarities[token] = similarity
            
    most_similar_words_with_similarity = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    return most_similar_words_with_similarity

def batch_find_sim_words(word_list, given_text):
    sim_list = []
    for word in word_list:
        try:
            sim_list += find_similar_words(word, given_text, w2v_model)
        except:
            sim_list += [(word, 1)]
    sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
    return sim_list

if __name__ == "__main__":
    theme_word = ["entrepreneurs", "professionals"]
    given_text = """Please create a series of promotional content for the Wolf Pack Coaching Program, a 12-week online sales training course targeted at aspiring entrepreneurs and sales professionals. The content should include persuasive and urgent emails, a compelling landing page text, engaging social media posts, descriptive product descriptions, and catchy promotional materials. The tone should create a sense of scarcity and exclusivity, emphasizing the urgency to join the program."""

    most_similar_words = batch_find_sim_words(theme_word, given_text)
    print(most_similar_words)
