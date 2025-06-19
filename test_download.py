from huggingface_hub import hf_hub_download
import pickle

pkl_path = hf_hub_download(
    repo_id="ShaikZo/embedding-cache",
    filename="embeddings.pkl"
)

print("Downloaded:", pkl_path)

with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

print("Loaded:", type(obj))
