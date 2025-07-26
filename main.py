import pickle
import asyncio
import uuid
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from app.data_loader import load_markdown_as_chunks, embed_chunks, mean_pooling
from app.rag import RAGSearch
from app.memory import conversation_memory
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
    title="Zo - Zohaib's JARVIS-like AI Assistant",
    description="An intelligent, conversational AI assistant powered by Google Gemini and GROQ with RAG capabilities. Like JARVIS, Zo provides natural, human-tailored responses about Zohaib Shaik's professional background, maintaining conversation context and delivering personalized interactions.",
    version="2.0.0"
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
DEFAULT_SIMILARITY_THRESHOLD = 0.80  # Increased for more relevant context
DEFAULT_TOP_K_CHUNKS = 3  # Reduced for more focused context
GROQ_MODEL_NAME = "llama-3.1-8b-instant" # Optimized for speed - best for JARVIS-like responses

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
            # A small test call to ensure connectivity (GROQ is synchronous, not async)
            try:
                _ = groq_client.chat.completions.create(
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

def create_resume_prompt(user_input: str, context_chunk: str, conversation_context: str = "", assistant_name: str = "Zo", is_first_query: bool = False) -> str:
    """
    Generates a JARVIS-like, human-tailored prompt for the AI assistant.
    Creates natural, conversational responses that are concise and directly answer the question.
    """
    # Refined JARVIS-like personality traits - more focused and human
    personality_traits = """
    You are Zo, Zohaib Shaik's AI assistant - think of yourself as a knowledgeable colleague who gives direct, helpful answers.
    
    PERSONALITY TRAITS:
    - Direct and conversational - answer the specific question asked
    - Confident but not overwhelming - share relevant details without excessive backstory
    - Human-like responses - speak naturally, not like a verbose encyclopedia
    - Focus on what the user actually wants to know
    - Only provide additional context if it directly relates to the question
    
    RESPONSE STYLE:
    - Answer the question first, then add brief relevant context if needed
    - Keep responses concise and focused - no unnecessary elaboration
    - Be natural and conversational, like a helpful colleague
    - Only mention achievements/projects when directly relevant to the question
    - Avoid repeating information unless specifically asked
    """

    base_instruction = f"""
    {personality_traits}

    {conversation_context}

    ---
    ZOHAIB'S PROFILE CONTEXT:
    {context_chunk}
    ---
    USER'S QUESTION: {user_input}
    ---
    
    INSTRUCTIONS:
    - Answer the specific question asked based on the profile context
    - Be concise and direct - don't overwhelm with unnecessary details
    - Use conversation history to avoid repeating information
    - If information isn't available, briefly state that and suggest related areas
    - Keep responses focused and human-like
    - Only elaborate when the question specifically asks for details
    """

    recruiter_keywords = ['opportunity', 'hiring', 'company', 'interview', 'recruiter', 'job', 'position', 'role', 'team', 'connect', 'hire', 'employment', 'candidate', 'vacancy', 'opening']
    is_recruiter_query = any(keyword in user_input.lower() for keyword in recruiter_keywords)
    
    greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
    is_greeting = any(keyword in user_input.lower() for keyword in greeting_keywords)

    intro_message = ""
    if is_first_query:
        if is_recruiter_query:
            intro_message = "Hi! I'm Zo, Zohaib's AI assistant. Happy to help with your inquiry about him."
        elif is_greeting:
            intro_message = "Hello! I'm Zo, Zohaib's AI assistant. What would you like to know about him?"
        else:
            intro_message = "Hi! I'm Zo, Zohaib's assistant. Let me help with your question."
    
    # Add an empty line if an intro message was added, for separation
    if intro_message:
        intro_message += "\n\n"

    additional_instruction = ""
    if is_recruiter_query:
        additional_instruction = """
        RECRUITER-FOCUSED RESPONSE:
        - Answer directly about what they're asking
        - Mention specific relevant achievements if asked
        - Keep it professional and focused
        - Provide contact info when appropriate: email: shaikzohaibgec@gmail.com, phone: +91 6281732166, LinkedIn: linkedin.com/in/zohaib-shaik-1a8877216
        """
    else:
        additional_instruction = """
        GENERAL RESPONSE GUIDELINES:
        - Answer the specific question asked
        - Keep responses focused and natural
        - Don't over-explain unless asked for details
        - If off-topic, briefly redirect: "I'm here to help with questions about Zohaib's professional background. What would you like to know?"
        """
    
    return intro_message + base_instruction + additional_instruction + "\n\nProvide a concise, natural response that directly answers the question:"


# --- Gemini Response Generation Function ---

async def generate_response_with_gemini(user_input: str, context_chunk: str, conversation_context: str, is_first_query: bool) -> str:
    """
    Generates a response using the Gemini model based on user input and retrieved context.
    Returns a special sentinel string if quota is exceeded.
    """
    global gemini_model
    if gemini_model is None:
        logger.error("Gemini model is not initialized. Cannot generate response with Gemini.")
        return ""

    prompt = create_resume_prompt(user_input, context_chunk, conversation_context, assistant_name="Zo", is_first_query=is_first_query)
    
    try:
        response = await gemini_model.generate_content_async(prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Slightly lower for more focused responses
                max_output_tokens=400,  # Reduced for more concise responses
                top_p=0.7,  # More focused sampling
                top_k=30,  # More focused top-k sampling
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

async def generate_response_with_groq(user_input: str, context_chunk: str, conversation_context: str, is_first_query: bool) -> str:
    """
    Generates a response using the GROQ model based on user input and retrieved context.
    """
    global groq_client
    if groq_client is None:
        logger.error("GROQ client is not initialized. Cannot generate response with GROQ.")
        return ""

    prompt_content = create_resume_prompt(user_input, context_chunk, conversation_context, assistant_name="Zo", is_first_query=is_first_query)
    
    try:
        # GROQ client methods are synchronous, not async
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Zo, Zohaib Shaik's AI assistant. Give direct, helpful answers without excessive details. Be conversational but concise - answer what's asked, don't over-explain."},
                {"role": "user", "content": prompt_content},
            ],
            model=GROQ_MODEL_NAME, # Use the defined model name here
            temperature=0.3,  # Lower for more focused responses
            max_tokens=400,  # Reduced for concise responses
            top_p=0.7,  # More focused sampling
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
            # GROQ client methods are synchronous, not async
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

@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    """
    Get a summary of the conversation session for analytics and context understanding.
    """
    summary = conversation_memory.get_session_summary(session_id)
    has_history = conversation_memory.has_conversation_history(session_id)
    
    return {
        "session_id": session_id,
        "has_history": has_history,
        "summary": summary,
        "total_turns": len(conversation_memory.conversations.get(session_id, [])),
    }

@app.post("/session/reset")
async def reset_session(req: Request):
    """
    Reset or clear a conversation session.
    """
    try:
        data = await req.json()
        session_id = data.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required.")
        
        if session_id in conversation_memory.conversations:
            del conversation_memory.conversations[session_id]
            logger.info(f"Reset session: {session_id}")
            return {"message": "Session reset successfully", "session_id": session_id}
        else:
            return {"message": "Session not found or already empty", "session_id": session_id}
    
    except Exception as e:
        logger.exception(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset session")

@app.post("/query")
async def handle_query(req: Request):
    """
    Handles incoming user queries with JARVIS-like conversation memory and context awareness.
    Provides natural, human-tailored responses with session continuity.
    """
    try:
        data = await req.json()
        user_input = data.get("input", "").strip()
        is_first_query = data.get("is_first_query", False)
        session_id = data.get("session_id", str(uuid.uuid4()))  # Generate session ID if not provided

        if not user_input:
            raise HTTPException(status_code=400, detail="Input query is required.")
        
        logger.info(f"Received query: '{user_input}', Session: {session_id}, Is First Query: {is_first_query}")

        # Get conversation context for continuity
        conversation_context = ""
        if not is_first_query and conversation_memory.has_conversation_history(session_id):
            conversation_context = conversation_memory.get_conversation_context(session_id, max_turns=2)

        retrieved_chunk = ""
        if rag and tokenizer_embedding_model_tuple: # Ensure RAG is fully initialized
            retrieved_chunk = rag.query(user_input, tokenizer_embedding_model_tuple, 
                                        top_k=DEFAULT_TOP_K_CHUNKS, 
                                        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)
        else:
             logger.warning("RAG system not fully available. Querying LLMs without resume context.")

        tasks = []
        task_to_api = {}  # Create mapping as we create tasks
        
        if gemini_model:
            gemini_task = asyncio.create_task(generate_response_with_gemini(user_input, retrieved_chunk, conversation_context, is_first_query), name="gemini_task")
            tasks.append(gemini_task)
            task_to_api[gemini_task] = "Gemini"  # Direct mapping
            logger.debug("Added Gemini task to processing queue")
            
        if groq_client:
            groq_task = asyncio.create_task(generate_response_with_groq(user_input, retrieved_chunk, conversation_context, is_first_query), name="groq_task")
            tasks.append(groq_task)
            task_to_api[groq_task] = "GROQ"  # Direct mapping
            logger.debug("Added GROQ task to processing queue")

        if not tasks:
            logger.error("Neither Gemini nor GROQ models are initialized. Cannot process query.")
            raise HTTPException(status_code=503, detail="AI assistant is not ready. No LLM available.")

        # Use asyncio.as_completed to get the first successful response
        chosen_response = ""
        api_source = "N/A"
        gemini_quota_hit = False
        completed_tasks = []
        
        logger.info(f"Task mapping created: {len(task_to_api)} tasks mapped")
        for task, api in task_to_api.items():
            logger.info(f"Task {id(task)} -> {api}")
        
        # Process responses as they complete (fastest first)
        for completed_task in asyncio.as_completed(tasks):
            try:
                logger.info(f"Processing completed task: {id(completed_task)}")
                res = await completed_task
                completed_tasks.append(completed_task)
                
                # Get the API source from our mapping
                current_api_source = task_to_api.get(completed_task, "Unknown")
                logger.info(f"Task {id(completed_task)} mapped to: {current_api_source}")
                
                if res == "GEMINI_QUOTA_EXCEEDED":
                    gemini_quota_hit = True
                    logger.warning("Gemini quota exceeded, trying next available response...")
                    continue # Don't consider this a valid response, but note the quota issue

                # Check if response is valid and not a "blocked" message
                if res and "i'm sorry, i cannot provide a response to that question." not in res.lower():
                    api_source = current_api_source
                    chosen_response = res
                    logger.info(f"Using first successful response from {api_source}")
                    # Cancel remaining tasks to save resources
                    for task in tasks:
                        if task not in completed_tasks and not task.done():
                            task.cancel()
                    break # Found a valid response, stop searching
                    
            except Exception as e:
                current_api_source = task_to_api.get(completed_task, "Unknown")
                logger.error(f"Task from {current_api_source} raised an exception: {e}")
                continue # Skip this response if it's an exception

        # If after checking all responses, nothing valid was found
        if not chosen_response:
            if gemini_quota_hit:
                logger.warning(f"Gemini quota hit and no other LLM provided a valid response for query: '{user_input}'")
                fallback_response = (
                    "I'm sorry, I'm currently experiencing high demand and my processing is temporarily limited. "
                    "However, I'm still here to help! You can reach Zohaib directly at shaikzohaibgec@gmail.com or "
                    "connect with him on LinkedIn. He'd be delighted to discuss opportunities personally."
                )
                return {"response": fallback_response, "session_id": session_id}
            else:
                logger.warning(f"No valid response generated from any LLM for query: '{user_input}'")
                fallback_response = (
                    "I apologize, but I'm having trouble generating a response right now. "
                    "Let me suggest trying a different approach to your question, or feel free to ask about "
                    "Zohaib's specific expertise in AI/ML, backend development, or his projects."
                )
                return {"response": fallback_response, "session_id": session_id}
        
        # Store the conversation for future context
        conversation_memory.add_conversation(
            session_id=session_id,
            user_input=user_input,
            assistant_response=chosen_response,
            context_used=retrieved_chunk[:200] if retrieved_chunk else ""
        )

        logger.info(f"Successfully generated response from {api_source} for query: '{user_input}', Session: {session_id}")
        return {
            "response": chosen_response, 
            "session_id": session_id,
            "api_source": api_source,
            "has_context": bool(retrieved_chunk)
        }
    
    except HTTPException as he:
        logger.warning(f"HTTP Exception caught: {he.detail}")
        raise he
    except Exception as e:
        logger.exception(f"Internal Server Error during query handling for input '{user_input if 'user_input' in locals() else 'N/A'}'.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Something went wrong while processing your request. Please try again.")

@app.get("/models/{api_source}")
async def get_model_status(api_source: str):
    """
    Check the status and available models for the specified API source.
    """
    if api_source == "gemini":
        if gemini_model is None:
            return {"status": "not initialized", "message": "Gemini model is not initialized"}
        try:
            # Use the correct method to list models from genai
            models = genai.list_models()
            model_names = [model.name for model in models]
            return {
                "status": "initialized", 
                "current_model": "gemini-1.5-flash-latest",
                "available_models": model_names # Limit to first 10 for readability
            }
        except Exception as e:
            logger.error(f"Error fetching Gemini models: {e}")
            return {"status": "error", "message": str(e)}
    
    elif api_source == "groq":
        if groq_client is None:
            return {"status": "not initialized", "message": "GROQ client is not initialized"}
        try:
            # Use direct API call to get GROQ models
            import requests
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                return {"status": "error", "message": "GROQ API key not available"}
            
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [model["id"] for model in models_data.get("data", [])]
                return {
                    "status": "initialized", 
                    "current_model": GROQ_MODEL_NAME,
                    "available_models": model_names
                }
            else:
                return {"status": "error", "message": f"API request failed with status {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching GROQ models: {e}")
            return {"status": "error", "message": str(e)}
    
    else:
        raise HTTPException(status_code=404, detail=f"API source '{api_source}' not found. Available: gemini, groq")


# --- Run Application ---
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))