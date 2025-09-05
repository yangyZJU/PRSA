from gensim.models import KeyedVectors
import numpy as np
import re
import scorers
from diff_module.topic_analysis import topic_identification
import nltk
import re
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google_news_related_words
import llm


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')


model_path = 'tool/GoogleNews-vectors-negative300.bin.gz'
sim_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
nlp = spacy.load('en_core_web_md')
matching_model = spacy.load("en_core_web_sm")

def clean_and_split(text):
    return re.sub(r'[^\w\d]', ' ', text).split()

def find_matching_phrases(config, input_data, output_data, prompt, fn):
    matching_phrases = []
    beam_phrases = []
    doc = matching_model(prompt)
    prompt_phrases = [chunk.text for chunk in doc.noun_chunks]
    if not re.search(r'\[.*?\]|\:|\=', input_data) and len(input_data.split()) <= 5:
        input_content = clean_and_split(input_data)
    else:
        input_content = re.findall(r'[:=|-]\s*(.+)', input_data)
    
    matches_dic = re.findall(r'\[([^\]]+)\]\s*[:=]\s*([^\n]+)|(\w[\w\s]*):\s*(.+)', input_data)

    result_dict = {}
    for match in matches_dic:
        if match[0] and match[1]:
            key = match[0].strip()
            value = match[1].strip()
        elif match[2] and match[3]:
            key = match[2].strip()
            value = match[3].strip()
        result_dict[key] = value

    for content in input_content:
        doc = matching_model(content)
        input_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        if input_phrases:
            matching_phrases.append(content)
            
            for input_per_phraase in input_phrases:
                vectorizer = TfidfVectorizer().fit_transform([input_per_phraase] + prompt_phrases)
                vectors = vectorizer.toarray()
                input_vector = vectors[0]
                similarities = cosine_similarity([input_vector], vectors[1:])[0]
                most_similar_index = np.argmax(similarities)
                most_similar_phrase = prompt_phrases[most_similar_index]
                beam_phrases.append(most_similar_phrase)
            
        else:
            matching_phrases.append(content)
    
    if len(beam_phrases) > 1:
        if config["beam_search"] == 1:
            beam_size = 1  
            f = max(int(len(beam_phrases) * config["alpha"]), 1)
            print("======Run selective beam search====== ... ...")
            best_beam = sequential_beam_search(config, input_data, beam_phrases, f, beam_size, prompt, output_data, fn)
        else:
            best_beam = beam_phrases
    else:
        best_beam = beam_phrases
    
    matching_phrases += best_beam
    return  matching_phrases, result_dict



def phrase_vector(phrase, model):
    """
    Compute a vector representation of a phrase by averaging the vectors of the words in the phrase.
    """
    words = phrase.split()
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(word_vectors, axis=0)

def are_synonyms(word1, word2, threshold=0.3):
    """
    Checks whether two words are synonyms.
    """
    vector1 = phrase_vector(word1, sim_model)
    vector2 = phrase_vector(word2, sim_model)
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if similarity >= threshold:
        return True
    else:
        return False


def update_gradient_dict(gradient, gradient_dict):
    for word in gradient:
        if word in gradient_dict:
            gradient_dict[word] += 1
        else:
            found_synonym = False

            if not found_synonym:
                gradient_dict[word] = 1

def filter_and_sort_dict(input_dict, filter_value=5):
    """
    Filters out items in the dictionary whose values are equal to `filter_value`,
    then sorts the remaining items in descending order by value.

    Args:
        input_dict (dict): The input dictionary to be filtered and sorted.
        filter_value (float, optional): The value to filter out. 
            By default, it is set to len(data) / 10, which retains the top 90% by confidence.

    Returns:
        dict: A new dictionary with filtered and sorted items (by value in descending order).
    """

    filtered_dict = {k: v for k, v in input_dict.items() if v > filter_value}

    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_dict = slice_dict(sorted_dict)
    return sorted_dict



def score_beam(beam, input_data, prompt, output_data, fn):
    scorer = scorers.MetricsScorer("semantic_similarity", 1)
    for word in beam:
        prompt = replace_token_with_placeholder(prompt, word)
    score_list = []
    
    pred_outputs = [fn(input_data, prompt) ]
    target_outputs = [output_data]
    sem_score = scorer.eval_pred_target("semantic_similarity", pred_outputs, target_outputs)
    score_list.append(sem_score)
    return np.mean(score_list)


def sequential_beam_search(config, input_data, vocabulary, f, beam_size, prompt, output_data, fn):
    beams = [(0, set())]
    res = []
    iteration = min(len(vocabulary), f)
    #print(iteration)
    score_interval = max(1, iteration // config["related_words_interval"])

    for i in range(0, iteration + 1, score_interval):
        new_beams = []

        current_length = min(i, iteration)
        current_beam = set(vocabulary[:current_length])

        current_score = score_beam(current_beam,input_data, prompt, output_data, fn)
        new_beams.append((current_score, current_beam))
        new_beams.sort(reverse=True)
        beams = new_beams[:beam_size]
        res += beams
    max_beam = max(res, key=lambda x: x[0])
    return list(max_beam[1])


def identify_related_words_google(config, keywords, input_data, text, output_data, fn):
    related_words_with_similarity = google_news_related_words.batch_find_sim_words(keywords, text)
    related_words_candidate =  [word for word, similarity in related_words_with_similarity if word.lower() not in config["theme"].lower()]

    if len(related_words_candidate) > 1:
        if config["beam_search"] == 1:
            beam_size = 1  
            f = max(int(len(related_words_candidate) * config["alpha"]), 1)
            print("======Run selective beam search====== ... ...")
            best_beam = sequential_beam_search(config, input_data, related_words_candidate, f, beam_size, text, output_data, fn)
        else:
            best_beam = related_words_candidate
    else:
        best_beam = related_words_candidate
    
    return best_beam

    


def identify_related_words(config, keywords, input_data, text, output_data, fn, similarity_threshold=0.5):
    
    doc_keywords = nlp(" ".join(keywords))
    doc_text = nlp(text)

    related_words_with_similarity = []
    for token_text in doc_text:
        for token_keyword in doc_keywords:
            similarity = token_text.similarity(token_keyword)
            if similarity > similarity_threshold:
                related_words_with_similarity.append((token_text.text, similarity))

    sorted_related_words = sorted(set(related_words_with_similarity), key=lambda x: x[1], reverse=True)
    related_words_candidate =  [word for word, similarity in sorted_related_words if word.lower() not in config["theme"].lower()]


    if len(related_words_candidate) > 1:
        if config["beam_search"] == 1:
            beam_size = 1  
            f = max(int(len(related_words_candidate) * config["alpha"]), 1)
            print("======Run selective beam search====== ... ...")
            best_beam = sequential_beam_search(config, input_data, related_words_candidate, f, beam_size, text, output_data, fn)
        else:
            best_beam = related_words_candidate
    else:
        best_beam = related_words_candidate
    
    return best_beam

def find_topic(config, input_data, output_data):
    theme = config["theme"]

    if not re.search(r'\[.*?\]|\:|\=', input_data) and len(input_data.split()) <= 5:
        return clean_and_split(input_data)
    else:
        matches = re.findall(r'[:=|-]\s*(.+)', input_data)
        #print(matches)
        if theme == "Code":
            documents = matches
        else:
            documents = matches

        topic_list = topic_identification(documents)
        cleaned_topics = [cleaned_word for topic in topic_list for cleaned_word in clean_and_split(topic)]
        return [word.lower() for word in cleaned_topics if theme.lower() not in word.lower()]



def parse_tagged_text(text, start_tag, end_tag):
    """ Parse text that is tagged with start and end tags."""
    texts = []
    while True:
        start_index = text.find(start_tag)
        if start_index == -1:
            break
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            break
        start_index += len(start_tag)
        texts.append(text[start_index:end_index].strip())
        text = text[end_index+len(end_tag):]
    return texts


def replace_token_with_placeholder(text, token, replace='{}'):
    words = text.split()
    token_escaped = re.escape(token)

    if token.endswith('y'):
        pattern = re.compile(rf'\b{token_escaped[:-1]}(y|ies)\b', re.IGNORECASE)
    elif token.endswith('f'):
        pattern = re.compile(rf'\b{token_escaped[:-1]}(f|ves)\b', re.IGNORECASE)
    elif token.endswith('fe'):
        pattern = re.compile(rf'\b{token_escaped[:-2]}(fe|ves)\b', re.IGNORECASE)
    else:
        pattern = re.compile(rf'\b{token_escaped}(s|es)?\b', re.IGNORECASE)

    matches = list(pattern.finditer(text))
    modified_text = text
    for match in reversed(matches):
        start, end = match.span()
        modified_text = modified_text[:start] + replace + modified_text[end:]

    return modified_text


def prompt_pruning(config, prompt, input_data, output_data, fn):
    prompt = mask_related_phrases(config, input_data, output_data, prompt, fn)
    prompt = mask_related_words(config, prompt, input_data, output_data, fn)
    return prompt


def prompt_pruning_google(config, prompt, input_data, output_data, fn):
    if config["pre_pruning"] == 1:
        prompt = llm.pre_pruning(input_data, prompt)
    prompt = mask_related_phrases(config, input_data, output_data, prompt, fn)
    prompt = mask_related_words_google(config, prompt, input_data, output_data, fn)
    return prompt

def prompt_pruning_phrase_level(config, prompt, input_data, output_data, fn):
    prompt = mask_related_phrases(config, input_data, output_data, prompt, fn)
    return prompt

def prompt_pruning_word_level(config, prompt, input_data, output_data, fn):
    prompt = mask_related_words(config, prompt, input_data, output_data, fn)
    return prompt

def prompt_pruning_word_level_google(config, prompt, input_data, output_data, fn):
    prompt = mask_related_words_google(config, prompt, input_data, output_data, fn)
    return prompt


def format_clean(text):
    
    result_text = re.sub(r'(\{\} )+', '{} ', text).strip()
    result_text = re.sub(r'(\{\})+', '{}', result_text)
    return result_text


def mask_related_phrases(config, input_data, output_data, prompt, fn):
    # mask strongly related phrases
    matching_phrases, result_dict = find_matching_phrases(config, input_data, output_data, prompt, fn)
    
    if config["theme"] == "Ads":
        config["theme"] = "advertisement"

    matching_phrases =  [word for word in matching_phrases if config["theme"].lower() not in word.lower()]

    for phrase in matching_phrases:
        if phrase.lower() not in stop_words:
            matching_keys = [key for key, value in result_dict.items() if phrase.lower() in value.lower()]
            if len(matching_keys)>0:
                prompt = replace_token_with_placeholder(prompt, phrase)
            else:
                prompt = replace_token_with_placeholder(prompt, phrase)
    return prompt


def mask_related_words_google(config, prompt, input_data, output_data, fn):

    raw_topic_list = find_topic(config, input_data, output_data)
    topic_list = [word for word in raw_topic_list if word.lower() not in stop_words]

    related_words = identify_related_words_google(config, topic_list, input_data, prompt, output_data, fn)
    filter_words = list(set(related_words))
    filter_words = [word for word in filter_words if word.lower() not in stop_words]
    
    for word in filter_words:
        prompt = replace_token_with_placeholder(prompt, word)

    return prompt




def mask_related_words(config, prompt, input_data, output_data, fn):

    raw_topic_list = find_topic(config, input_data, output_data)
    topic_list = [word for word in raw_topic_list if word.lower() not in stop_words]

    related_words = identify_related_words(config, topic_list, input_data, prompt, output_data, fn)
    filter_words = list(set(related_words))
    filter_words = [word for word in filter_words if word.lower() not in stop_words]

    for word in filter_words:
        prompt = replace_token_with_placeholder(prompt, word)
    
    return prompt


def slice_dict(original_dict, slice_percentage=50):

    slice_count = int(len(original_dict) * slice_percentage / 100)
    sliced_dict = dict(list(original_dict.items())[:slice_count])

    return sliced_dict
