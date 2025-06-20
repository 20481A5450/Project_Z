import pickle
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from app.data_loader import load_markdown_as_chunks, embed_chunks
from app.rag import RAGSearch
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging
import uvicorn # Add uvicorn import for running the app

# Trying to make the app better without latency issues, cold starts, or other common issues

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Zohaib's AI Resume Assistant",
    description="An AI assistant powered by Google Gemini and RAG to answer questions about Zohaib Shaik's resume.",
    version="1.0.0"
)

# Configure CORS to allow only your frontend (GitHub Pages)
# It's good that you had "*" for local testing, but revert to specific origins for production.
origins = [
    # "https://20481A5450.github.io",  # Replace with your actual GitHub Pages URL
    "http://localhost:3000", # Example for local React/Vue/Angular dev server
    "http://127.0.0.1:3000", # Another common local dev server address
    "http://localhost:8000", # If your frontend is served from here
    "http://127.0.0.1:8000",
    "*" # Keep during development, but tighten for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Consider limiting to ["GET", "POST"] for production APIs
    allow_headers=["*"],
)

# Configure Gemini - Moved to a dedicated function for clarity and error handling
gemini_model = None # Initialize as None, will be set in on_startup

# Define constants
EMBEDDINGS_REPO = "ShaikZo/embedding-cache"
EMBEDDINGS_FILENAME = "embeddings.pkl"
RESUME_DATA_PATH = "data/data.md" # Define the path to your markdown resume

# Global variables for RAG components
rag = None
tokenizer_embedding_model_tuple = None

# --- Startup Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """
    Initializes critical components (Gemini, Embeddings, RAG) on application startup.
    This ensures that the app is ready to serve requests once started.
    """
    logger.info("Application startup initiated.")

    # 1. Configure Gemini
    global gemini_model
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=google_api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        # Test Gemini model to ensure connectivity
        # This is a lightweight way to check if the API key is valid and model is accessible
        _ = gemini_model.generate_content("hello", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
        logger.info("Google Gemini model configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure or connect to Google Gemini model: {e}")
        # Depending on criticality, you might want to exit here or run in a degraded state
        # For an AI assistant, Gemini is critical, so failing fast might be better.
        # raise RuntimeError(f"Critical error: Gemini model initialization failed: {e}")

    # 2. Load and embed markdown resume data
    global rag, tokenizer_embedding_model_tuple
    try:
        # Attempt to download embeddings first
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        pkl_path = None
        if hf_token: # Only try to download if token is available
            try:
                logger.info(f"Attempting to download {EMBEDDINGS_FILENAME} from Hugging Face Hub...")
                # Download to a temporary path or ensure it's saved where `embed_chunks` expects
                pkl_path = hf_hub_download(
                    repo_id=EMBEDDINGS_REPO,
                    filename=EMBEDDINGS_FILENAME,
                    token=hf_token
                )
                logger.info(f"Downloaded embeddings to: {pkl_path}")
            except Exception as e:
                logger.warning(f"Failed to download cached embeddings from Hugging Face Hub: {e}. Will generate locally if needed.")
                # If download fails, pkl_path remains None, embed_chunks will try to generate/load locally

        # Pass the path where embed_chunks should look for/save the pickle
        # If pkl_path is None (download failed), embed_chunks will use its default or specified path
        tokenizer_embedding_model_tuple, embeddings, chunks = embed_chunks(chunks=load_markdown_as_chunks(RESUME_DATA_PATH), embed_path=pkl_path or f"data/{EMBEDDINGS_FILENAME}")

        if embeddings is None or chunks is None:
            raise RuntimeError("Embeddings or chunks could not be loaded/generated.")

        rag = RAGSearch(embeddings, chunks)
        logger.info("RAGSearch system initialized successfully.")

    except Exception as e:
        logger.exception("Failed to initialize RAGSearch system (embeddings, chunks).")
        # This is a critical component for RAG. If it fails, the app won't function as intended.
        # Consider raising an error to prevent the app from starting in a broken state.
        # For deployment, a health check would catch this later, but preventing startup is safer.
        # raise RuntimeError(f"Critical error: RAGSearch initialization failed: {e}")
    
    logger.info("Application startup complete.")


# --- Gemini Response Generation Function ---

# Moved generate_response_with_gemini directly into main.py (or keep in a dedicated module if preferred)
def generate_response_with_gemini(user_input, context_chunk):
    """
    Generates a response using the Gemini model based on user input and retrieved context.
    Includes refined persona and instruction for recruiters/casual questions.
    """
    if gemini_model is None:
        logger.error("Gemini model is not initialized.")
        return "I'm sorry, my AI brain is not fully online yet. Please try again in a moment."

    prompt = f"""
    You are a professional, friendly AI assistant representing a developer named Zohaib Shaik — or Zo for short. You are named after him, you're Zo.
    Your job is to respond to incoming questions based *strictly* and *only* on the resume information provided below.
    If the answer is not explicitly stated or directly inferable from the 'Resume Info', you must politely state that the information is not available in Zohaib's resume. Do not make up information.

    If the user appears to be a recruiter, hiring manager, or expresses interest in an 'opportunity', 'hiring', 'company', or 'interview', acknowledge their professional intent. Offer to notify Zohaib directly or suggest they contact him via his provided contact information (if available in the resume context). For example: "It sounds like you're interested in a potential opportunity. I can confirm [answer based on resume]. Would you like me to share Zohaib's contact information or pass along your message?"

    If the question is casual (e.g., 'Who are you?', 'How are you doing?', 'Tell me a joke'), answer in a generic way like a person do in real life.

    ---
    Resume Info:
    {context_chunk}
    ---
    User’s Question: {user_input}
    ---
    Your response:
    """
    
    try:
        response = gemini_model.generate_content(prompt, 
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Lower temperature for more factual, less creative responses
                max_output_tokens=500 # Limit output length to prevent rambling
            )
        )
        # Check if the response actually contains text
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            logger.warning(f"Gemini returned an empty or invalid response for query: {user_input}")
            return "" # Explicitly return empty string if no valid text
    except Exception as e:
        logger.error(f"Error generating Gemini response for input '{user_input[:50]}...': {e}")
        return "" # Return empty string on error to trigger fallback message

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """
    Provides a health check endpoint to monitor the status of critical components.
    """
    status = {"status": "healthy", "components": {}}
    errors = []

    # Check Gemini Model
    if gemini_model is not None:
        try:
            # Perform a very quick, minimal call to verify connectivity
            _ = gemini_model.generate_content("ping", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
            status["components"]["gemini"] = "ok"
        except Exception as e:
            status["components"]["gemini"] = "error"
            errors.append(f"Gemini connectivity failed: {e}")
            logger.error(f"Health check: Gemini connectivity error: {e}")
    else:
        status["components"]["gemini"] = "not initialized"
        errors.append("Gemini model not initialized.")

    # Check RAG System
    if rag is not None and rag.chunks is not None and len(rag.chunks) > 0:
        status["components"]["resume_data"] = f"{len(rag.chunks)} chunks loaded"
        status["components"]["embeddings"] = f"{rag.embeddings.shape[0]} vectors"
    else:
        status["components"]["resume_data"] = "not loaded/empty"
        status["components"]["embeddings"] = "not loaded/empty"
        errors.append("RAG data (chunks/embeddings) not loaded or empty.")

    if errors:
        status["status"] = "degraded" if len(errors) < 2 else "unhealthy" # More granular status
        status["message"] = "; ".join(errors)

    return status

@app.post("/query")
async def handle_query(req: Request):
    """
    Handles incoming user queries, performs RAG, and returns a Gemini-generated response.
    Includes robust error handling and a fallback message.
    """
    try:
        if rag is None or tokenizer_embedding_model_tuple is None:
            logger.error("RAG system or embedding model not initialized during query.")
            raise HTTPException(status_code=503, detail="AI assistant is not ready. Please try again in a moment.")

        data = await req.json()
        user_input = data.get("input", "").strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="Input query is required.")
        
        logger.info(f"Received query: '{user_input}'")

        # Use a slightly lower top_k to encourage more precise context and prevent overwhelming the LLM
        # And leverage the similarity_threshold added in rag.py
        retrieved_chunk = rag.query(user_input, tokenizer_embedding_model_tuple, top_k=3, similarity_threshold=0.75)
        
        # If no relevant chunk is found by RAG (due to similarity_threshold)
        if not retrieved_chunk:
            logger.info(f"No relevant resume info found for query: '{user_input}'")
            return {
                "response": (
                    "I couldn’t find information directly related to that in Zohaib’s resume. "
                    "Please try asking a different question or rephrasing your current one. "
                    "Would you like me to pass your message along to him?"
                )
            }

        logger.debug(f"Retrieved chunk for '{user_input[:50]}...': {retrieved_chunk}") # Use debug for large outputs

        response = generate_response_with_gemini(user_input, retrieved_chunk)

        # Enhance the "no answer" detection from Gemini
        # Check for common phrases indicating inability to answer from context
        no_answer_phrases = ["I couldn't find", "doesn't mention", "not stated", "information is not available", "not present in the resume"]
        if not response or any(phrase in response.lower() for phrase in no_answer_phrases):
            logger.info(f"Gemini indicated no relevant info for query: '{user_input}'")
            return {
                "response": (
                    "I’m sorry, I couldn’t find a relevant answer in Zohaib’s resume based on the information I have. "
                    "Would you like me to pass your message along to him?"
                )
            }

        logger.info(f"Successfully generated response for query: '{user_input}'")
        return {"response": response}
    
    except HTTPException as he:
        logger.warning(f"HTTP Exception: {he.detail}")
        raise he # Re-raise HTTPExceptions
    except Exception as e:
        logger.exception(f"Internal Server Error during query handling for input '{user_input if 'user_input' in locals() else 'N/A'}'.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Something went wrong while processing your request. Please try again.")

@app.get("/models")
def list_gemini_models():
    """
    Lists available Gemini models. Useful for debugging and understanding model capabilities.
    """
    if gemini_model is None:
        logger.error("Attempted to list models before Gemini was initialized.")
        raise HTTPException(status_code=503, detail="Gemini model not initialized.")
    try:
        models = genai.list_models()
        model_list = []
        for m in models:
            # Filter for models that support text generation if you only care about that
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                model_list.append({
                    "name": m.name,
                    "display_name": getattr(m, "display_name", "N/A"),
                    "generation_methods": getattr(m, "supported_generation_methods", "N/A")
                })
        logger.info(f"Listed {len(model_list)} Gemini models.")
        return {"models": model_list}
    except Exception as e:
        logger.exception("Error fetching Gemini models list.")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# --- Run Application ---
if __name__ == "__main__":
    # Ensure the 'data' directory exists for local embeddings cache
    os.makedirs("data", exist_ok=True)
    # Use uvicorn.run for production-ready server
    # host="0.0.0.0" makes it accessible from outside localhost
    # port is taken from environment variable or defaults to 7860
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))