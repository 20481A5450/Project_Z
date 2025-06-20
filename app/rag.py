from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from app.data_loader import mean_pooling # Assuming mean_pooling remains in data_loader
import logging

logger = logging.getLogger(__name__)

class RAGSearch:
    def __init__(self, embeddings: np.ndarray, chunks: list[str]):
        """
        Initializes the RAGSearch system with pre-computed embeddings and their corresponding chunks.
        
        Args:
            embeddings (np.ndarray): A 2D numpy array where each row is the embedding of a chunk.
            chunks (list[str]): A list of strings, where each string is a text chunk, corresponding to the embeddings.
        """
        if embeddings is None or chunks is None or len(chunks) == 0:
            # More specific error message
            raise ValueError("Embeddings and chunks must be provided, be non-empty, and valid types for RAGSearch initialization.")
        
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise TypeError("Embeddings must be a 2D numpy array.")
        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
            raise TypeError("Chunks must be a list of strings.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) must match number of chunks ({len(chunks)}).")

        self.embeddings = embeddings
        self.chunks = chunks
        logger.info(f"RAGSearch initialized with {len(self.chunks)} chunks and {self.embeddings.shape[0]} embeddings.")

    def query(self, user_input: str, model_tuple: tuple, top_k: int = 5, similarity_threshold: float = 0.75) -> str:
        """
        Performs a RAG query: embeds the user input, finds similar chunks,
        and returns a concatenated string of the most relevant chunks.
        
        Args:
            user_input (str): The user's query string.
            model_tuple (tuple): A tuple containing the tokenizer and embedding model.
            top_k (int): The number of top relevant chunks to retrieve *after* filtering by threshold.
            similarity_threshold (float): Minimum cosine similarity to consider a chunk relevant.
                                          Chunks below this threshold will be excluded.
        Returns:
            str: A concatenated string of relevant chunks, or an empty string if none are relevant.
        """
        tokenizer, model = model_tuple
        
        if tokenizer is None or model is None:
            logger.error("Tokenizer or embedding model not initialized in RAGSearch query.")
            return "" # Return empty string if embedding model is not ready

        # Ensure model is on the correct device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy() # Move to CPU before numpy

        # Ensure query_embedding is in a 2D array for cosine_similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Calculate cosine similarities between query and all stored embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)
        
        # Get indices and scores of results, sorted by similarity in descending order
        sorted_indices = np.argsort(similarities[0])[::-1]
        
        relevant_chunks_with_scores = []
        for i in sorted_indices:
            score = similarities[0][i]
            if score >= similarity_threshold:
                # Store the actual chunk string and its score
                relevant_chunks_with_scores.append((self.chunks[i], score))
            else:
                # Since indices are sorted by similarity, we can stop early if score is below threshold
                break 
        
        if not relevant_chunks_with_scores:
            logger.info(f"No chunks found above similarity threshold ({similarity_threshold}) for query: '{user_input[:50]}...'")
            return "" # Return empty string if no relevant chunks meet the threshold

        # Limit to the top_k relevant chunks found
        final_relevant_chunks = [chunk for chunk, score in relevant_chunks_with_scores[:top_k]]
        
        logger.info(f"Retrieved {len(final_relevant_chunks)} relevant chunks (top {top_k} from threshold-filtered results).")
        
        # Join chunks with a clear separator for the LLM
        return "\n---\n".join(final_relevant_chunks)