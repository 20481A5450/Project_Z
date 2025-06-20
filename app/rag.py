from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from app.data_loader import mean_pooling # Assuming mean_pooling remains in data_loader
import logging

logger = logging.getLogger(__name__)

class RAGSearch:
    def __init__(self, embeddings, chunks):
        if embeddings is None or chunks is None or len(chunks) == 0:
            raise ValueError("Embeddings and chunks must be provided and not empty for RAGSearch.")
        self.embeddings = embeddings
        self.chunks = chunks
        logger.info(f"RAGSearch initialized with {len(self.chunks)} chunks and {self.embeddings.shape[0]} embeddings.")

    def query(self, user_input, model_tuple, top_k=3, similarity_threshold=0.75):
        """
        Performs a RAG query: embeds the user input, finds similar chunks,
        and returns a concatenated string of the most relevant chunks.
        
        Args:
            user_input (str): The user's query string.
            model_tuple (tuple): A tuple containing the tokenizer and embedding model.
            top_k (int): The number of top relevant chunks to retrieve.
            similarity_threshold (float): Minimum cosine similarity to consider a chunk relevant.
                                          Chunks below this threshold will be excluded.
        Returns:
            str: A concatenated string of relevant chunks, or an empty string if none are relevant.
        """
        tokenizer, model = model_tuple
        
        # Ensure model is on the correct device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy() # Move to CPU

        # Ensure embeddings are in a 2D array for cosine_similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.embeddings)
        
        # Get indices and scores of top_k results
        # np.argsort returns indices that would sort an array
        # [::-1] reverses it to get descending order
        # [:top_k] slices to get only the top_k
        sorted_indices = np.argsort(similarities[0])[::-1]
        
        relevant_chunks_with_scores = []
        for i in sorted_indices:
            score = similarities[0][i]
            if score >= similarity_threshold:
                relevant_chunks_with_scores.append((self.chunks[i], score))
            else:
                # Since indices are sorted by similarity, we can stop early
                break 
        
        if not relevant_chunks_with_scores:
            logger.info(f"No chunks found above similarity threshold ({similarity_threshold}) for query: '{user_input[:50]}...'")
            return "" # Return empty string if no relevant chunks

        # You can consider re-ranking here if you have a re-ranker model.
        # For now, we'll just take the top_k from the initial similarity search
        # and limit to the actual top_k requested (after filtering by threshold)
        final_relevant_chunks = [chunk for chunk, score in relevant_chunks_with_scores[:top_k]]
        
        logger.info(f"Retrieved {len(final_relevant_chunks)} relevant chunks (top {top_k}, threshold {similarity_threshold}).")
        
        # Join chunks with a clear separator for the LLM
        return "\n---\n".join(final_relevant_chunks)