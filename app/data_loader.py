from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os

def load_markdown_as_chunks(path, max_chunk_length=400):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    raw_chunks = re.split(r'\n#{2,3} ', content)

    clean_chunks = []
    current_chunk = ""

    for raw in raw_chunks:
        if len(raw.strip()) == 0:
            continue
        lines = raw.strip().splitlines()
        header = lines[0] if lines else ""
        body = " ".join(lines[1:]) if len(lines) > 1 else ""
        combined = f"{header.strip()}: {body.strip()}"
        if len(combined) <= max_chunk_length:
            clean_chunks.append(combined)
        else:
            sub_chunks = re.split(r'\n|\.\s+', body)
            current = header + ": "
            for part in sub_chunks:
                if len(current) + len(part) <= max_chunk_length:
                    current += part + " "
                else:
                    clean_chunks.append(current.strip())
                    current = header + ": " + part + " "
            if current.strip():
                clean_chunks.append(current.strip())
    return clean_chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
            torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def embed_chunks(chunks, embed_path="data/embeddings.pkl"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        model = AutoModel.from_pretrained("thenlper/gte-small")

        if os.path.exists(embed_path):
            with open(embed_path, "rb") as f:
                return (tokenizer, model), *pickle.load(f)

        encoded_input = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        embeddings = embeddings.detach().numpy()

        with open(embed_path, "wb") as f:
            pickle.dump((embeddings, chunks), f)

        return (tokenizer, model), embeddings, chunks
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None, None, None
