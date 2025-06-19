from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.data_loader import load_markdown_as_chunks, embed_chunks
from app.rag import RAGSearch
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging

# Load environment variables from .env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS to allow only your frontend (GitHub Pages)
app = FastAPI()
origins = [
    # "https://20481A5450.github.io",  # Replace with your actual GitHub Pages URL
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Download and load precomputed embeddings from Hugging Face
EMBEDDINGS_REPO = "ShaikZo/zo-embeddings"
EMBEDDINGS_FILENAME = "embeddings.pkl"

try:
    logger.info("Downloading embeddings.pkl from Hugging Face Hub...")
    pkl_path = hf_hub_download(repo_id=EMBEDDINGS_REPO, filename=EMBEDDINGS_FILENAME)

    with open(pkl_path, "rb") as f:
        (tokenizer, embedding_model), embeddings, chunks = pickle.load(f)

    rag = RAGSearch(embeddings, chunks)
    logger.info("✅ Embeddings loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to download or load embeddings: {e}")
    raise RuntimeError("Could not initialize embeddings from Hugging Face Hub.")


# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Gemini response generation function
def generate_response_with_gemini(user_input, context_chunk):
    prompt = f"""
        You are a professional, friendly AI assistant representing a software developer named Zohaib Shaik — or Zo for short, You're named after him you're Zo.
        Your job is to respond to incoming questions based only on the resume info provided below.
        If the user appears to be a recruiter or hiring manager (based on words like 'opportunity', 'hiring', 'company', or 'interview'), acknowledge that professionally and offer to notify Zohaib or share more information.
        If the question is casual (e.g. 'Who are you?'), introduce yourself as Zohaib's assistant and answer with helpful, polite context.
        Never invent facts. If something is not mentioned in the resume, say so politely.
        Resume Info:{context_chunk}
        User’s Question: {user_input}
        Your response:
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Load and embed markdown resume
chunks = load_markdown_as_chunks("data/data.md")
(tokenizer, embedding_model), embeddings, chunks = embed_chunks(chunks)
rag = RAGSearch(embeddings, chunks)

@app.get("/health")
def health_check():
    try:
        Verify critical components
        if not hasattr(gemini_model, 'generate_content'):
            raise Exception("Gemini model not initialized")
        
        # Verify data loaded
        if len(rag.chunks) == 0:
            raise Exception("No resume data loaded")
            
        return {"status": "healthy", "components": {
            "gemini": "ok",
            "resume_data": f"{len(rag.chunks)} chunks loaded",
            "embeddings": f"{rag.embeddings.shape} vectors"
        }}
        # pass
    except Exception as e:
        return {"status": "error", "message": str(e)}       

@app.post("/query")
async def handle_query(req: Request):
    try:
        data = await req.json()
        user_input = data.get("input", "")
        if not user_input:
            raise HTTPException(status_code=400, detail="Input is required")

        # print(f"User input: {user_input}")
        retrieved_chunk = rag.query(user_input, (tokenizer, embedding_model), top_k=5)
        # print(f"Retrieved chunk: {retrieved_chunk}")

        response = generate_response_with_gemini(user_input, retrieved_chunk)

        if not response or "doesn't mention" in response.lower():
            return {
                "response": (
                    "I’m sorry, I couldn’t find a relevant answer in Zohaib’s resume. "
                    "Would you like me to pass your message along to him?"
                )
            }

        # print(f"Gemini Response: {response}")
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/models")
def list_gemini_models():
    try:
        models = genai.list_models()
        model_list = []
        for m in models:
            model_list.append({
                "name": m.name,
                "display_name": getattr(m, "display_name", "N/A"),
                "generation_methods": getattr(m, "supported_generation_methods", "N/A")
            })
        return {"models": model_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
