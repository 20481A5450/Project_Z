from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from app.data_loader import mean_pooling

class RAGSearch:
    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks

    def query(self, user_input, model_tuple, top_k=5):
        tokenizer, model = model_tuple
        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask']).numpy()

        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        relevant_text = "\n\n".join([self.chunks[i] for i in top_indices])
        return relevant_text


