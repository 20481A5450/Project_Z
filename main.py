import pickle
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from app.data_loader import load_markdown_as_chunks, embed_chunks, mean_pooling
from app.rag import RAGSearch
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging
import uvicorn
import numpy as np
import groq

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Zohaib's AI Resume Assistant",
    description="An AI assistant powered by Google Gemini and RAG to answer questions about Zohaib Shaik's resume, with dynamic API routing.",
    version="1.0.0"
)

# Configure CORS to allow specific origins for your frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://20481A5450.github.io",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for critical components, initialized as None
gemini_model = None
groq_client = None
rag = None
tokenizer_embedding_model_tuple = None

# Define constants
EMBEDDINGS_REPO = "ShaikZo/embedding-cache"
EMBEDDINGS_FILENAME = "embeddings.pkl"
RESUME_DATA_PATH = "data/data.md"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_TOP_K_CHUNKS = 5
GROQ_MODEL_NAME = "llama3-8b-8192" # Define the GROQ model name here

# --- Startup Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """
    Initializes critical components (Gemini, GROQ, Embeddings, RAG) on application startup.
    """
    logger.info("Application startup initiated.")

    # 1. Configure Gemini
    global gemini_model
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY environment variable not set. Gemini model will not be initialized.")
            gemini_model = None
        else:
            genai.configure(api_key=google_api_key)
            try:
                # A small test call to ensure connectivity
                _ = await genai.GenerativeModel("gemini-1.5-flash-latest").generate_content_async("test", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
                gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                logger.info("Google Gemini model configured successfully.")
            except Exception as gemini_conn_err:
                logger.error(f"Failed to connect to Google Gemini API with provided key: {gemini_conn_err}. Gemini model will not be available.")
                gemini_model = None
    except Exception as e:
        logger.exception(f"Unexpected error during Gemini configuration: {e}")
        gemini_model = None

    # 2. Configure GROQ Client
    global groq_client
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY environment variable not set. GROQ API client will not be initialized.")
            groq_client = None
        else:
            groq_client = groq.Groq(api_key=groq_api_key)
            # A small test call to ensure connectivity
            try:
                _ = await groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model=GROQ_MODEL_NAME, # Use the defined model name here
                    max_tokens=1
                )
                logger.info(f"GROQ API client initialized successfully using model: {GROQ_MODEL_NAME}.")
            except Exception as groq_conn_err:
                logger.error(f"Failed to connect to GROQ API with provided key or model '{GROQ_MODEL_NAME}': {groq_conn_err}. GROQ API will not be available.")
                groq_client = None
    except Exception as e:
        logger.exception(f"Unexpected error during GROQ configuration: {e}")
        groq_client = None

    # 3. Load and embed markdown resume data
    global rag, tokenizer_embedding_model_tuple
    try:
        local_embeddings_cache_path = os.path.join("data", EMBEDDINGS_FILENAME)
        os.makedirs("data", exist_ok=True)

        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        downloaded_pkl_path = None
        if hf_token:
            try:
                logger.info(f"Attempting to download {EMBEDDINGS_FILENAME} from Hugging Face Hub...")
                downloaded_pkl_path = hf_hub_download(
                    repo_id=EMBEDDINGS_REPO,
                    filename=EMBEDDINGS_FILENAME,
                    token=hf_token
                )
                logger.info(f"Downloaded embeddings to: {downloaded_pkl_path}")
            except Exception as e:
                logger.warning(f"Failed to download cached embeddings from Hugging Face Hub: {e}. Will attempt to use/generate local cache.")
        
        chunks_from_markdown = load_markdown_as_chunks(RESUME_DATA_PATH)
        
        actual_embed_path_for_func = downloaded_pkl_path or local_embeddings_cache_path

        tokenizer_embedding_model_tuple, embeddings_data, processed_chunks_for_rag = \
            embed_chunks(chunks_from_markdown, embed_path=actual_embed_path_for_func)

        if embeddings_data is None or processed_chunks_for_rag is None:
            raise RuntimeError("Embeddings or chunks could not be loaded/generated by embed_chunks.")

        rag = RAGSearch(embeddings_data, processed_chunks_for_rag)
        logger.info("RAGSearch system initialized successfully.")

    except Exception as e:
        logger.exception("Failed to initialize RAGSearch system (embeddings, chunks). AI assistant will operate in a degraded mode.")
        rag = None
        tokenizer_embedding_model_tuple = None
    
    logger.info("Application startup complete.")


# --- Prompt Generation Functions ---

def create_resume_prompt(user_input: str, context_chunk: str, assistant_name: str = "Zo", is_first_query: bool = False) -> str:
    """
    Generates a friendly and professional prompt for the AI assistant.
    Adjusts persona based on user query and available context.
    Conditionally adds introductory message based on is_first_query flag.
    """
    base_instruction = f"""
    You are {assistant_name}, Zohaib Shaik's friendly and professional AI assistant.
    Your main goal is to help users by answering questions strictly based on the "Resume Information" provided.
    If you cannot find the answer explicitly in the "Resume Information", please politely and genuinely state that the information isn't available in Zohaib's resume. Do not invent details.

    ---
    Resume Information:
    {context_chunk}
    ---
    User's Question: {user_input}
    ---
    """

    recruiter_keywords = ['opportunity', 'hiring', 'company', 'interview', 'recruiter', 'job', 'position', 'role', 'team', 'connect']
    is_recruiter_query = any(keyword in user_input.lower() for keyword in recruiter_keywords)

    intro_message = ""
    if is_first_query:
        if is_recruiter_query:
            intro_message = "Hello there! As Zohaib's assistant, I'd be happy to help with that. Regarding Zohaib's resume..."
        else:
            intro_message = "Hello! I'm Zo, Zohaib's AI assistant, here to help you navigate his resume. Regarding your question..."
    
    # Add an empty line if an intro message was added, for separation
    if intro_message:
        intro_message += "\n\n"

    additional_instruction = ""
    if is_recruiter_query:
        additional_instruction = f"""
        After providing the answer based on the resume, gently offer to connect them with Zohaib or suggest they use his contact details (phone: +91 6281732166, email: shaikzohaibgec@gmail.com, LinkedIn: linkedin.com/in/zohaib-shaik-1a8877216).
        Prioritize providing the direct contact details if asked, or offer to pass a message if preferred.
        """
    else:
        additional_instruction = """
        For general or casual questions (e.g., about yourself as an AI, greetings, or off-topic questions):
        Gently guide the conversation back to topics related to Zohaib's professional background or resume.
        If a question is completely irrelevant and cannot be related to the resume, politely decline to answer,
        e.g., "That's an interesting question, but my purpose is to answer questions about Zohaib's professional profile. Is there anything else about his resume I can help you with?"
        """
    
    return intro_message + base_instruction + additional_instruction + "\nYour helpful and concise response:"


# --- Gemini Response Generation Function ---

async def generate_response_with_gemini(user_input: str, context_chunk: str, is_first_query: bool) -> str:
    """
    Generates a response using the Gemini model based on user input and retrieved context.
    Returns a special sentinel string if quota is exceeded.
    """
    global gemini_model
    if gemini_model is None:
        logger.error("Gemini model is not initialized. Cannot generate response with Gemini.")
        return ""

    prompt = create_resume_prompt(user_input, context_chunk, assistant_name="Zo", is_first_query=is_first_query)
    
    try:
        response = await gemini_model.generate_content_async(prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=500
            ),
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.text.strip()
        elif response and response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Gemini response blocked for reason: {response.prompt_feedback.block_reason} for input: '{user_input}'")
            return "I'm sorry, I cannot provide a response to that question."
        else:
            logger.warning(f"Gemini returned an empty or invalid response for query: '{user_input}'")
            return ""

    except Exception as e:
        error_message = str(e)
        if "429 You exceeded your current quota" in error_message or "Quota exceeded" in error_message:
            logger.error(f"Gemini quota exceeded for input '{user_input[:50]}...'.")
            return "GEMINI_QUOTA_EXCEEDED" # Special sentinel value
        logger.error(f"Error generating Gemini response for input '{user_input[:50]}...': {e}")
        return "" # Default empty string for other errors

# --- GROQ Response Generation Function ---

async def generate_response_with_groq(user_input: str, context_chunk: str, is_first_query: bool) -> str:
    """
    Generates a response using the GROQ model based on user input and retrieved context.
    """
    global groq_client
    if groq_client is None:
        logger.error("GROQ client is not initialized. Cannot generate response with GROQ.")
        return ""

    prompt_content = create_resume_prompt(user_input, context_chunk, assistant_name="Zo", is_first_query=is_first_query)
    
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Zo, Zohaib Shaik's friendly and professional AI assistant. You answer questions strictly based on the provided resume information."},
                {"role": "user", "content": prompt_content},
            ],
            model=GROQ_MODEL_NAME, # Use the defined model name here
            temperature=0.2,
            max_tokens=500,
        )
        
        if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content:
            return chat_completion.choices[0].message.content.strip()
        else:
            logger.warning(f"GROQ returned an empty or invalid response for query: '{user_input}'")
            return ""

    except Exception as e:
        logger.error(f"Error generating GROQ response for input '{user_input[:50]}...': {e}")
        return ""


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

    # Check GROQ Client
    if groq_client is not None:
        try:
            _ = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model=GROQ_MODEL_NAME, # Use the defined model name here
                max_tokens=1
            )
            status["components"]["groq"] = "ok"
        except Exception as e:
            status["components"]["groq"] = "error"
            errors.append(f"GROQ connectivity failed: {e}")
            logger.error(f"Health check: GROQ connectivity error: {e}")
    else:
        status["components"]["groq"] = "not initialized"
        errors.append("GROQ client not initialized.")

    # Check RAG System
    if rag is not None and rag.chunks is not None and len(rag.chunks) > 0 and rag.embeddings is not None:
        status["components"]["resume_data"] = f"{len(rag.chunks)} chunks loaded"
        status["components"]["embeddings"] = f"{rag.embeddings.shape[0]} vectors"
    else:
        status["components"]["resume_data"] = "not loaded/empty"
        status["components"]["embeddings"] = "not loaded/empty"
        errors.append("RAG data (chunks/embeddings) not loaded or empty.")

    if errors:
        status["status"] = "degraded" if len(errors) < 2 else "unhealthy"
        status["message"] = "; ".join(errors)

    return status

@app.post("/query")
async def handle_query(req: Request):
    """
    Handles incoming user queries, performs RAG, and returns a response from the faster LLM.
    """
    try:
        data = await req.json()
        user_input = data.get("input", "").strip()
        is_first_query = data.get("is_first_query", False)

        if not user_input:
            raise HTTPException(status_code=400, detail="Input query is required.")
        
        logger.info(f"Received query: '{user_input}', Is First Query: {is_first_query}")

        retrieved_chunk = ""
        if rag and tokenizer_embedding_model_tuple: # Ensure RAG is fully initialized
            retrieved_chunk = rag.query(user_input, tokenizer_embedding_model_tuple, 
                                        top_k=DEFAULT_TOP_K_CHUNKS, 
                                        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)
        else:
             logger.warning("RAG system not fully available. Querying LLMs without resume context.")

        tasks = []
        if gemini_model:
            tasks.append(asyncio.create_task(generate_response_with_gemini(user_input, retrieved_chunk, is_first_query), name="gemini_task"))
        if groq_client:
            tasks.append(asyncio.create_task(generate_response_with_groq(user_input, retrieved_chunk, is_first_query), name="groq_task"))

        if not tasks:
            logger.error("Neither Gemini nor GROQ models are initialized. Cannot process query.")
            raise HTTPException(status_code=503, detail="AI assistant is not ready. No LLM available.")

        # Run all tasks concurrently and collect results and exceptions
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        chosen_response = ""
        api_source = "N/A"
        gemini_quota_hit = False

        # Check for the first valid response in the order of task creation (or completion)
        for i, res in enumerate(responses):
            if isinstance(res, Exception):
                task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else "unknown_task"
                logger.error(f"Task '{task_name}' raised an exception: {res}")
                continue # Skip this response if it's an exception

            if res == "GEMINI_QUOTA_EXCEEDED":
                gemini_quota_hit = True
                continue # Don't consider this a valid response, but note the quota issue

            # Check if response is valid and not a "blocked" message
            if res and "i'm sorry, i cannot provide a response to that question." not in res.lower():
                if tasks[i].get_name() == "gemini_task":
                    api_source = "Gemini"
                elif tasks[i].get_name() == "groq_task":
                    api_source = "GROQ"
                chosen_response = res
                break # Found a valid response, stop searching

        # If after checking all responses, nothing valid was found
        if not chosen_response:
            if gemini_quota_hit:
                logger.warning(f"Gemini quota hit and no other LLM provided a valid response for query: '{user_input}'")
                return {
                    "response": (
                        "I'm sorry, Zohaib's AI assistant is currently experiencing high demand. "
                        "The Gemini API quota has been reached. Please try again later, or contact Zohaib directly via his email at shaikzohaibgec@gmail.com."
                    )
                }
            else:
                logger.warning(f"No valid response generated from any LLM for query: '{user_input}'")
                return {
                    "response": (
                        "I'm truly sorry, but I couldn't generate a helpful response for that right now. "
                        "Perhaps try rephrasing your question or asking something else about Zohaib's resume. "
                        "If you're a recruiter, I'd be happy to share Zohaib's contact details!"
                    )
                }
        
        logger.info(f"Successfully generated response from {api_source} for query: '{user_input}'")
        return {"response": chosen_response}
    
    except HTTPException as he:
        logger.warning(f"HTTP Exception caught: {he.detail}")
        raise he
    except Exception as e:
        logger.exception(f"Internal Server Error during query handling for input '{user_input if 'user_input' in locals() else 'N/A'}'.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Something went wrong while processing your request. Please try again.")

# --- Run Application ---
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))