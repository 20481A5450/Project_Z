from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os
import re
import logging

logger = logging.getLogger(__name__)

def load_markdown_as_chunks(path, max_chunk_length=400, chunk_overlap_ratio=0.1):
    """
    Loads markdown content and splits it into semantic chunks with optional overlap.
    Chunks are designed to respect structural elements like headers.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by level 2 or 3 headers (## or ###) to get initial semantic sections
    # This creates more meaningful "raw" chunks
    raw_sections = re.split(r'\n(##|###) ', content)
    
    # The first element might be empty or content before the first header,
    # so we need to handle it. Also, re.split includes the delimiter itself,
    # so we pair them up.
    processed_sections = []
    if raw_sections and not raw_sections[0].startswith('#'): # If content before first header
        if raw_sections[0].strip():
            processed_sections.append(raw_sections[0].strip())
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, len(raw_sections), 2):
        if i + 1 < len(raw_sections):
            header_level = raw_sections[i] # e.g., '##' or '###'
            header_title_and_body = raw_sections[i+1]
            processed_sections.append(f"{header_level} {header_title_and_body.strip()}")
        elif raw_sections[i].strip(): # Handle the very last part if it exists
             processed_sections.append(raw_sections[i].strip())


    clean_chunks = []
    
    for section_content in processed_sections:
        # Extract header for consistent prefixing, assuming the format "## Header: Body"
        # This regex tries to capture the header part up to the first newline or colon
        header_match = re.match(r'^(##|###)\s*([^:\n]+)([:\n])?', section_content)
        current_header = ""
        if header_match:
            current_header = header_match.group(2).strip()
            body_content = section_content[header_match.end():].strip()
        else: # No clear header found, treat the whole thing as body
            body_content = section_content.strip()

        # If the entire section is within limits, add it directly
        if len(section_content) <= max_chunk_length:
            clean_chunks.append(section_content.strip())
        else:
            # For longer sections, split by sentence or significant punctuation
            # Adding an overlap for better context
            sentences = re.split(r'(?<=[.!?])\s+|\n', body_content)
            
            current_chunk_parts = []
            current_chunk_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Add a header prefix to each sub-chunk derived from a section
                prefixed_sentence = f"{current_header}: {sentence}" if current_header else sentence

                if current_chunk_length + len(prefixed_sentence) + 1 <= max_chunk_length: # +1 for space
                    current_chunk_parts.append(prefixed_sentence)
                    current_chunk_length += len(prefixed_sentence) + 1
                else:
                    if current_chunk_parts:
                        clean_chunks.append(" ".join(current_chunk_parts).strip())
                    
                    # Start new chunk with overlap
                    overlap_size = int(max_chunk_length * chunk_overlap_ratio)
                    overlap_parts = []
                    
                    # Add parts from the end of the previous chunk to the start of the new one
                    temp_len = 0
                    for part in reversed(current_chunk_parts):
                        if temp_len + len(part) <= overlap_size:
                            overlap_parts.insert(0, part)
                            temp_len += len(part)
                        else:
                            break
                    
                    current_chunk_parts = overlap_parts + [prefixed_sentence]
                    current_chunk_length = sum(len(p) + 1 for p in current_chunk_parts) - 1 # Recalculate length


            if current_chunk_parts: # Add any remaining parts
                clean_chunks.append(" ".join(current_chunk_parts).strip())
    
    logger.info(f"Loaded {len(clean_chunks)} chunks from {path}")
    return clean_chunks

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the token embeddings to get a single vector for the sequence.
    """
    token_embeddings = model_output[0]  # First element is last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def embed_chunks(chunks, embed_path="data/embeddings.pkl"):
    """
    Loads the tokenizer and model, then generates and caches embeddings for the chunks.
    Uses 'thenlper/gte-small' as a good all-around embedding model.
    """
    try:
        # Using a specific, well-regarded embedding model
        model_name = "thenlper/gte-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Embedding model '{model_name}' loaded on {device}")

        # Check if cached embeddings exist
        if os.path.exists(embed_path):
            try:
                with open(embed_path, "rb") as f:
                    logger.info(f"Loading cached embeddings from {embed_path}")
                    cached_data = pickle.load(f)
                    # Ensure the loaded data matches the expected format (embeddings, chunks)
                    if isinstance(cached_data, tuple) and len(cached_data) == 2 and \
                       isinstance(cached_data[0], np.ndarray) and isinstance(cached_data[1], list):
                        # Optionally, verify if cached chunks match current chunks
                        # This can be more complex if content changes frequently
                        # For simplicity, we trust the cache for now.
                        if len(cached_data[1]) == len(chunks) and all(c1 == c2 for c1, c2 in zip(cached_data[1], chunks)):
                             logger.info("Cached embeddings found and match current chunks. Using cached.")
                             return (tokenizer, model), cached_data[0], cached_data[1]
                        else:
                            logger.warning("Cached chunks do not match current chunks. Re-embedding data.")
                    else:
                        logger.warning("Cached embeddings file format is unexpected. Re-embedding data.")
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}. Re-embedding data.")
        
        logger.info("Generating new embeddings...")
        # Generate embeddings if cache is invalid or doesn't exist
        encoded_input = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy() # Move to CPU before numpy

        with open(embed_path, "wb") as f:
            pickle.dump((embeddings, chunks), f)
        logger.info(f"Embeddings generated and saved to {embed_path}")

        return (tokenizer, model), embeddings, chunks
    except Exception as e:
        logger.exception("Error during embedding generation or loading.")
        return None, None, None