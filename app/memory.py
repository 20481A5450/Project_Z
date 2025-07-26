import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    user_input: str
    assistant_response: str
    timestamp: float
    context_used: str
    session_id: str

class ConversationMemory:
    """
    Manages conversation history for JARVIS-like contextual awareness.
    Stores recent conversations to provide continuity and reference previous discussions.
    """
    
    def __init__(self, max_history: int = 10, session_timeout: int = 3600):
        """
        Initialize conversation memory system.
        
        Args:
            max_history: Maximum number of conversation turns to remember per session
            session_timeout: Session timeout in seconds (1 hour default)
        """
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.max_history = max_history
        self.session_timeout = session_timeout
    
    def add_conversation(self, session_id: str, user_input: str, assistant_response: str, context_used: str = ""):
        """Add a conversation turn to memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        turn = ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            timestamp=time.time(),
            context_used=context_used,
            session_id=session_id
        )
        
        self.conversations[session_id].append(turn)
        
        # Keep only the most recent conversations
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
        
        logger.info(f"Added conversation turn for session {session_id}")
    
    def get_conversation_context(self, session_id: str, max_turns: int = 2) -> str:
        """
        Get recent conversation context for continuity without overwhelming detail.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Concise conversation context string
        """
        if session_id not in self.conversations:
            return ""
        
        # Clean expired sessions
        self._clean_expired_sessions()
        
        recent_turns = self.conversations[session_id][-max_turns:]
        
        if not recent_turns:
            return ""
        
        context_parts = ["=== RECENT CONVERSATION ==="]
        
        for i, turn in enumerate(recent_turns):
            context_parts.append(f"\nPrevious Q: {turn.user_input}")
            # Truncate previous response to avoid overwhelming context
            truncated_response = turn.assistant_response[:150] + "..." if len(turn.assistant_response) > 150 else turn.assistant_response
            context_parts.append(f"Your answer: {truncated_response}")
        
        context_parts.append("\n=== END CONVERSATION ===")
        context_parts.append("Build on this context naturally, don't repeat the same information.")
        
        return "\n".join(context_parts)
    
    def _clean_expired_sessions(self):
        """Remove expired sessions to manage memory"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, turns in self.conversations.items():
            if not turns:
                expired_sessions.append(session_id)
                continue
                
            # Check if the most recent turn is older than session timeout
            if current_time - turns[-1].timestamp > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversations[session_id]
            logger.info(f"Cleaned expired session: {session_id}")
    
    def has_conversation_history(self, session_id: str) -> bool:
        """Check if a session has any conversation history"""
        return session_id in self.conversations and len(self.conversations[session_id]) > 0
    
    def get_session_summary(self, session_id: str) -> str:
        """Get a summary of the session for analytical purposes"""
        if session_id not in self.conversations:
            return "No conversation history"
        
        turns = self.conversations[session_id]
        if not turns:
            return "No conversation history"
        
        topics = []
        for turn in turns:
            # Simple topic extraction based on keywords
            if any(keyword in turn.user_input.lower() for keyword in ['project', 'experience', 'skill']):
                topics.append("professional background")
            elif any(keyword in turn.user_input.lower() for keyword in ['contact', 'hire', 'recruiter']):
                topics.append("recruitment inquiry")
            elif any(keyword in turn.user_input.lower() for keyword in ['education', 'college', 'degree']):
                topics.append("educational background")
        
        unique_topics = list(set(topics))
        return f"Session with {len(turns)} turns covering: {', '.join(unique_topics) if unique_topics else 'general discussion'}"

# Global memory instance
conversation_memory = ConversationMemory()
