import pickle
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from app.data_loader import load_markdown_as_chunks, embed_chunks, mean_pooling # Ensure mean_pooling is imported if used elsewhere
from app.rag import RAGSearch
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging
import uvicorn
import numpy as np

# Modifications in code for better response and issues which are solved now

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

# Configure CORS to allow specific origins for your frontend
# IMPORTANT: For production, change "*" to your actual GitHub Pages URL(s)
origins = [  # Replace with your actual GitHub Pages URL
    "http://localhost:3000",      # For local React/Vue/Angular dev server
    "http://127.0.0.1:3000",      # Another common local dev server address
    "http://localhost:8000",      # If your frontend or testing tool is on this port
    "http://127.0.0.1:8000",
    # Add any other origins your frontend might be hosted on in production
    # Remove "*" once you have all production origins listed for security
    "*" # Keep during development for flexibility, but tighten for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Consider limiting to ["GET", "POST"] for production APIs
    allow_headers=["*"],
)

# Global variables for critical components, initialized as None
gemini_model = None
rag = None
tokenizer_embedding_model_tuple = None

# Define constants (can be moved to a separate config.py for larger projects)
EMBEDDINGS_REPO = "ShaikZo/embedding-cache"
EMBEDDINGS_FILENAME = "embeddings.pkl"
RESUME_DATA_PATH = "data/data.md"
DEFAULT_SIMILARITY_THRESHOLD = 0.75 # Good starting point, tune as needed
DEFAULT_TOP_K_CHUNKS = 5 # Changed from 3 to 5 based on your observation for better retrieval

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
            logger.error("GOOGLE_API_KEY environment variable not set. Gemini model will not be initialized.")
            gemini_model = None
            # Do NOT raise error here if you want the app to start in a degraded state
            # but queries will fail if Gemini is uninitialized.
        else:
            genai.configure(api_key=google_api_key)
            # A small test call to ensure connectivity
            try:
                # Use a very minimal generation to quickly check API connectivity and key validity
                _ = genai.GenerativeModel("gemini-1.5-flash-latest").generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
                gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                logger.info("Google Gemini model configured successfully.")
            except Exception as gemini_conn_err:
                logger.error(f"Failed to connect to Google Gemini API with provided key: {gemini_conn_err}. Gemini model will not be available.")
                gemini_model = None

    except Exception as e:
        logger.exception(f"Unexpected error during Gemini configuration: {e}")
        gemini_model = None # Ensure it's None if configuration itself fails

    # 2. Load and embed markdown resume data
    global rag, tokenizer_embedding_model_tuple
    try:
        # Determine the local path for the embeddings pickle
        local_embeddings_cache_path = os.path.join("data", EMBEDDINGS_FILENAME)
        os.makedirs("data", exist_ok=True) # Ensure 'data' directory exists for local caching

        # Attempt to download embeddings from Hugging Face Hub first
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        downloaded_pkl_path = None
        if hf_token:
            try:
                logger.info(f"Attempting to download {EMBEDDINGS_FILENAME} from Hugging Face Hub...")
                # hf_hub_download caches to ~/.cache/huggingface by default
                downloaded_pkl_path = hf_hub_download(
                    repo_id=EMBEDDINGS_REPO,
                    filename=EMBEDDINGS_FILENAME,
                    token=hf_token
                )
                logger.info(f"Downloaded embeddings to: {downloaded_pkl_path}")
            except Exception as e:
                logger.warning(f"Failed to download cached embeddings from Hugging Face Hub: {e}. Will attempt to use/generate local cache.")
        
        # Load raw chunks from your markdown file
        chunks_from_markdown = load_markdown_as_chunks(RESUME_DATA_PATH)
        
        # Pass the path where embed_chunks should look for/save the pickle.
        # Prioritize downloaded path, otherwise use local 'data/' path.
        actual_embed_path_for_func = downloaded_pkl_path or local_embeddings_cache_path

        tokenizer_embedding_model_tuple, embeddings_data, processed_chunks_for_rag = \
            embed_chunks(chunks_from_markdown, embed_path=actual_embed_path_for_func)

        if embeddings_data is None or processed_chunks_for_rag is None:
            raise RuntimeError("Embeddings or chunks could not be loaded/generated by embed_chunks.")

        rag = RAGSearch(embeddings_data, processed_chunks_for_rag)
        logger.info("RAGSearch system initialized successfully.")

    except Exception as e:
        logger.exception("Failed to initialize RAGSearch system (embeddings, chunks). AI assistant will operate in a degraded mode.")
        # Set rag to None to indicate it's not ready, handle this in query endpoint
        rag = None
        tokenizer_embedding_model_tuple = None
    
    logger.info("Application startup complete.")


# --- Gemini Response Generation Function ---

def generate_response_with_gemini(user_input: str, context_chunk: str) -> str:
    """
    Generates a response using the Gemini model based on user input and retrieved context.
    Includes refined persona and instruction for recruiters/casual questions.
    """
    global gemini_model # Use global model variable
    if gemini_model is None:
        logger.error("Gemini model is not initialized. Cannot generate response.")
        return "" # Return empty string to trigger fallback message

    # Refined prompt for clarity and conciseness
    prompt = f"""
    You are Zo, Zohaib Shaik's friendly and professional AI assistant.
    
    Your primary goal is to answer user questions using *only* the "Resume Info" provided.
    If the answer is not explicitly present or directly inferable from the "Resume Info", you must politely state that the information is not available in Zohaib's resume. Do not make up information.

    - **For Recruiters/Hiring Managers (indicated by words like 'opportunity', 'hiring', 'company', 'interview'):** Acknowledge their professional intent. Offer to notify Zohaib directly or suggest they contact him via his provided contact information (if available in the resume context). Example: "It sounds like you're interested in a potential opportunity. Based on Zohaib's resume, [answer]. Would you like me to share Zohaib's contact information or pass along your message?"

    - **For Casual Questions (e.g., 'Who are you?', 'How are you doing?', 'Tell me a joke'):** Introduce yourself as Zohaib's assistant, Zo, and respond in a friendly, helpful, and engaging manner. Politely steer the conversation back to topics related to Zohaib's resume or professional background.

    ---
    Resume Info:
    {context_chunk}
    ---
    User's Question: {user_input}
    ---
    Your response:
    """
    
    try:
        response = gemini_model.generate_content(prompt, 
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Lower temperature for more factual, less creative responses
                max_output_tokens=500 # Limit output length to prevent rambling
            ),
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Check if the response actually contains text and is not blocked
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.text.strip()
        elif response and response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Gemini response blocked for reason: {response.prompt_feedback.block_reason} for input: '{user_input}'")
            return "I'm sorry, I cannot provide a response to that question."
        else:
            logger.warning(f"Gemini returned an empty or invalid response for query: '{user_input}'")
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
            # Use a short timeout to prevent blocking if Gemini API is slow
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
    if rag is not None and rag.chunks is not None and len(rag.chunks) > 0 and rag.embeddings is not None:
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
        # Check if core components are ready to serve queries
        if rag is None or tokenizer_embedding_model_tuple is None or gemini_model is None:
            logger.error("AI assistant not fully initialized. Cannot process query.")
            raise HTTPException(status_code=503, detail="AI assistant is not ready. Please try again in a moment.")

        data = await req.json()
        user_input = data.get("input", "").strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="Input query is required.")
        
        logger.info(f"Received query: '{user_input}'")

        # Perform RAG search
        # Using configured DEFAULT_TOP_K_CHUNKS and DEFAULT_SIMILARITY_THRESHOLD
        retrieved_chunk = rag.query(user_input, tokenizer_embedding_model_tuple, 
                                    top_k=DEFAULT_TOP_K_CHUNKS, 
                                    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)
        
        # If no relevant chunk is found by RAG (due to similarity_threshold or no data)
        if not retrieved_chunk:
            logger.info(f"No relevant resume info found by RAG for query: '{user_input}' (threshold applied).")
            return {
                "response": (
                    "I couldn’t find information directly related to that in Zohaib’s resume. "
                    "Please try asking a different question or rephrasing your current one. "
                    "Would you like me to pass your message along to him?"
                )
            }

        logger.debug(f"Retrieved chunk for '{user_input[:50]}...': {retrieved_chunk}") # Use debug for large outputs

        # Generate response using Gemini
        response = generate_response_with_gemini(user_input, retrieved_chunk)

        # Enhance the "no answer" detection from Gemini's response
        no_answer_phrases = [
            "i couldn't find", "doesn't mention", "not stated", 
            "information is not available", "not present in the resume",
            "i don't have information", "not provided in the resume"
        ]
        if not response or any(phrase in response.lower() for phrase in no_answer_phrases):
            logger.info(f"Gemini indicated no relevant info (or blocked) for query: '{user_input}'")
            return {
                "response": (
                    "I’m sorry, I couldn’t find a relevant answer in Zohaib’s resume based on the information I have. "
                    "Would you like me to pass your message along to him?"
                )
            }

        logger.info(f"Successfully generated response for query: '{user_input}'")
        return {"response": response}
    
    except HTTPException as he:
        logger.warning(f"HTTP Exception caught: {he.detail}")
        raise he # Re-raise HTTPExceptions
    except Exception as e:
        # Catch any other unexpected errors and log them with traceback
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
    # Ensure the 'data' directory exists for local embeddings cache before uvicorn starts
    # This also helps with the Hugging Face Space persistent storage for `data/embeddings.pkl`
    os.makedirs("data", exist_ok=True)
    
    # Use uvicorn.run for production-ready server
    # host="0.0.0.0" makes it accessible from outside localhost
    # port is taken from environment variable or defaults to 7860
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))