from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sbert_model = sbert_model.to(device)

def calculate_similarity_sbert(text1, text2):
    embeddings = sbert_model.encode([text1, text2], convert_to_tensor=True, device=device)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()



if __name__ == '__main__':
    target = """List and describe the Top 5 Historical Adventures based on [Adventure Description] and [Historical Periods of Interest]. For each adventure, provide a detailed account of the activities, the historical context, and the unique aspects that make it a compelling experience. Highlight how each adventure allows travelers to immerse themselves in the chosen historical periods, offering insights into the culture, events, and lifestyle of those times.
    """
    stolen = """List and describe the Top 5 Historical Adventures based on [Adventure Description] and [Historical Periods of Interest]. For each adventure, provide a detailed account of the activities, the historical context, and the unique aspects that make it a compelling experience. Highlight how each adventure allows travelers to immerse themselves in the chosen historical periods, offering insights into the culture, events, and lifestyle of those times.
    """
    similarity_score = calculate_similarity_sbert(target, stolen)
    print(f"similarity_score: {similarity_score}")



