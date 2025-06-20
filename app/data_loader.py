import os
import re
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np  # Ensure this is imported for numpy array handling in pickle
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

# Assuming your "chunks" are just strings.
# If you decide to introduce a Chunk object with metadata later,
# you'll need to adapt this to work with Chunk instances.
# For now, we're treating chunks as list[str]

def load_markdown_as_chunks(path: str, max_chunk_length: int = 400, chunk_overlap_ratio: float = 0.1) -> list[str]:
    """
    Loads markdown content from a file, splits it into semantic chunks,
    and returns a list of chunk contents (strings).
    Attempts to preserve headings and adds overlap for context.
    """
    logger.info(f"Loading markdown from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by level 2 or 3 headers (## or ###) to get initial semantic sections
    raw_sections = re.split(r'\n(##|###) ', content)
    
    processed_sections = []
    # Handle content before the first header if it exists
    if raw_sections and not raw_sections[0].startswith('#') and raw_sections[0].strip():
        processed_sections.append(raw_sections[0].strip())
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, len(raw_sections), 2):
        if i + 1 < len(raw_sections):
            header_level = raw_sections[i] # e.g., '##' or '###'
            header_title_and_body = raw_sections[i+1]
            processed_sections.append(f"{header_level} {header_title_and_body.strip()}")
        elif raw_sections[i].strip(): # Handle the very last part if it exists without a subsequent header
             processed_sections.append(raw_sections[i].strip())


    clean_chunks = []
    
    for section_content in processed_sections:
        header_match = re.match(r'^(##|###)\s*([^:\n]+)([:\n])?', section_content)
        current_header = ""
        body_content = section_content.strip()

        if header_match:
            current_header = header_match.group(2).strip()
            # Extract body content by removing the matched header part
            body_content = section_content[header_match.end():].strip()

        # If the entire section is within limits, add it directly
        if len(section_content) <= max_chunk_length:
            clean_chunks.append(section_content.strip())
        else:
            # For longer sections, split by sentence or significant punctuation
            sentences = re.split(r'(?<=[.!?])\s+|\n', body_content) # Split by sentence-ending punctuation or newline
            
            current_chunk_parts = []
            current_chunk_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Add a header prefix to each sub-chunk derived from a section
                prefixed_sentence = f"{current_header}: {sentence}" if current_header else sentence

                # Check if adding the next sentence exceeds max_chunk_length
                if current_chunk_length + len(prefixed_sentence) + (1 if current_chunk_parts else 0) <= max_chunk_length: # +1 for space
                    current_chunk_parts.append(prefixed_sentence)
                    current_chunk_length += len(prefixed_sentence) + (1 if current_chunk_parts else 0)
                else:
                    # Current chunk is full, add it to clean_chunks
                    if current_chunk_parts:
                        clean_chunks.append(" ".join(current_chunk_parts).strip())
                    
                    # Start new chunk with overlap
                    overlap_size = int(max_chunk_length * chunk_overlap_ratio)
                    overlap_parts = []
                    
                    # Add parts from the end of the previous chunk to the start of the new one
                    temp_len = 0
                    for part in reversed(current_chunk_parts):
                        if temp_len + len(part) + (1 if overlap_parts else 0) <= overlap_size:
                            overlap_parts.insert(0, part) # Insert at beginning to maintain order
                            temp_len += len(part) + (1 if overlap_parts else 0)
                        else:
                            break
                    
                    current_chunk_parts = overlap_parts + [prefixed_sentence]
                    current_chunk_length = sum(len(p) + (1 if i > 0 else 0) for i, p in enumerate(current_chunk_parts)) # Recalculate length

            # Add any remaining parts in current_chunk_parts
            if current_chunk_parts:
                clean_chunks.append(" ".join(current_chunk_parts).strip())
    
    logger.info(f"Loaded {len(clean_chunks)} chunks from {path}")
    return clean_chunks

def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs mean pooling on the token embeddings to get a single vector for the sequence.
    """
    token_embeddings = model_output[0]  # First element is last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def embed_chunks(chunks_content: list[str], model_name: str = "thenlper/gte-small", embed_path: str = "data/embeddings.pkl") -> tuple[tuple, np.ndarray, list[str]]:
    """
    Generates or loads embeddings for text chunk contents (strings) using a specified Hugging Face model.
    Caches embeddings to a pickle file for faster subsequent loads.
    Returns (tokenizer, model), embeddings, original_chunks_content.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Embedding model '{model_name}' will use device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.error(f"Failed to load embedding model or tokenizer '{model_name}': {e}")
        return (None, None), None, None

    # Check if cached embeddings exist
    if os.path.exists(embed_path):
        try:
            logger.info(f"Loading cached embeddings from {embed_path}")
            with open(embed_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Expecting cached_data to be a tuple of (embeddings_array, list_of_strings_representing_chunks)
            if isinstance(cached_data, tuple) and len(cached_data) == 2 and \
               isinstance(cached_data[0], np.ndarray) and isinstance(cached_data[1], list):
                
                cached_embeddings, cached_chunks_content = cached_data
                
                # Basic consistency check: ensure the number of chunks matches
                # and loaded content are indeed strings. A deeper check would compare content.
                if len(cached_chunks_content) == len(chunks_content) and \
                   all(isinstance(c, str) for c in cached_chunks_content):
                    logger.info("Cached embeddings loaded successfully and match current data length.")
                    return (tokenizer, model), cached_embeddings, chunks_content
                else:
                    logger.warning("Cached data consistency check failed (e.g., chunk count mismatch or wrong type). Re-embedding.")
            else:
                logger.warning("Cached data format is unexpected. Re-embedding.")

        except Exception as e:
            logger.error(f"Error loading cached embeddings from {embed_path}: {e}. Re-embedding data.")
            # Continue to generate new embeddings if loading fails

    # If no cache or cache load failed, generate new embeddings
    logger.info("Generating new embeddings...")
    embeddings_list = []
    batch_size = 32 # Adjust based on available memory and performance
    
    # Process chunks in batches
    for i in range(0, len(chunks_content), batch_size):
        batch_chunks = chunks_content[i:i+batch_size]
        encoded_input = tokenizer(batch_chunks, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
        embeddings_list.append(batch_embeddings)
    
    embeddings_array = np.vstack(embeddings_list) # Combine all batches into a single numpy array

    # Save generated embeddings and the original chunk contents (strings) for future use
    try:
        # Save the embeddings array and the list of chunk content strings
        with open(embed_path, 'wb') as f:
            pickle.dump((embeddings_array, chunks_content), f) # <--- THIS IS THE KEY FIX
        logger.info(f"Embeddings generated and saved to {embed_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings to cache at {embed_path}: {e}")

    return (tokenizer, model), embeddings_array, chunks_content