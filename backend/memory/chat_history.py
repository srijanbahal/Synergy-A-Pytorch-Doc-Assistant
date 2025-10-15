"""
Conversation memory management for maintaining chat history and context.
"""
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class ConversationMemory:
    """Manages conversation history and context for multi-turn conversations."""
    
    def __init__(self, max_history: int = 10, max_context_length: int = 4000):
        self.max_history = max_history
        self.max_context_length = max_context_length
        
        # In-memory storage for conversation history
        # In production, this could be replaced with Redis or database
        self.conversations: Dict[str, deque] = {}
        
        # Conversation metadata
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Conversation memory initialized")
    
    def add_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_history)
            self.conversation_metadata[session_id] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "message_count": 0,
                "topics": set(),
                "entities": set()
            }
        
        # Add timestamp to message
        message["timestamp"] = time.time()
        
        # Add to conversation
        self.conversations[session_id].append(message)
        
        # Update metadata
        metadata = self.conversation_metadata[session_id]
        metadata["last_activity"] = time.time()
        metadata["message_count"] += 1
        
        # Extract topics and entities from message
        if message.get("type") == "user":
            self._extract_topics_and_entities(session_id, message.get("content", ""))
        
        logger.debug(f"Added message to session {session_id}")
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self.conversations:
            return []
        
        history = list(self.conversations[session_id])
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_conversation_context(self, session_id: str, max_length: Optional[int] = None) -> str:
        """Get formatted conversation context for LLM."""
        if session_id not in self.conversations:
            return ""
        
        history = self.get_conversation_history(session_id)
        
        if not history:
            return ""
        
        context_parts = []
        current_length = 0
        max_length = max_length or self.max_context_length
        
        # Build context from recent messages
        for message in reversed(history):
            if message.get("type") == "user":
                user_content = message.get("content", "")
                context_parts.insert(0, f"User: {user_content}")
                current_length += len(user_content)
            elif message.get("type") == "assistant":
                assistant_content = message.get("content", "")
                context_parts.insert(0, f"Assistant: {assistant_content}")
                current_length += len(assistant_content)
            
            # Stop if we exceed max length
            if current_length > max_length:
                break
        
        return "\n".join(context_parts)
    
    def summarize_conversation(self, session_id: str) -> Dict[str, Any]:
        """Generate a summary of the conversation."""
        if session_id not in self.conversations:
            return {"summary": "No conversation found"}
        
        metadata = self.conversation_metadata[session_id]
        history = self.get_conversation_history(session_id)
        
        # Basic statistics
        user_messages = [msg for msg in history if msg.get("type") == "user"]
        assistant_messages = [msg for msg in history if msg.get("type") == "assistant"]
        
        # Extract common topics
        topics = list(metadata.get("topics", set()))
        entities = list(metadata.get("entities", set()))
        
        # Time span
        if history:
            start_time = history[0].get("timestamp", 0)
            end_time = history[-1].get("timestamp", 0)
            duration = end_time - start_time
        else:
            duration = 0
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "topics": topics,
            "entities": entities,
            "duration_seconds": duration,
            "created_at": metadata.get("created_at", 0),
            "last_activity": metadata.get("last_activity", 0)
        }
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            del self.conversation_metadata[session_id]
            logger.info(f"Cleared conversation for session {session_id}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.conversations:
            return {"exists": False}
        
        metadata = self.conversation_metadata[session_id]
        history = self.get_conversation_history(session_id)
        
        return {
            "exists": True,
            "message_count": len(history),
            "created_at": metadata.get("created_at", 0),
            "last_activity": metadata.get("last_activity", 0),
            "topics": list(metadata.get("topics", set())),
            "entities": list(metadata.get("entities", set()))
        }
    
    def _extract_topics_and_entities(self, session_id: str, content: str) -> None:
        """Extract topics and entities from user message content."""
        metadata = self.conversation_metadata[session_id]
        
        # Simple topic extraction (could be enhanced with NLP)
        topics = metadata.get("topics", set())
        entities = metadata.get("entities", set())
        
        # Extract PyTorch-related terms
        pytorch_terms = [
            "tensor", "neural network", "model", "training", "optimization",
            "loss", "gradient", "backpropagation", "activation", "layer",
            "convolution", "pooling", "dropout", "batch normalization"
        ]
        
        content_lower = content.lower()
        for term in pytorch_terms:
            if term in content_lower:
                topics.add(term)
        
        # Extract function/class names (simple pattern matching)
        import re
        function_pattern = r'\b(\w+\.\w+)\b'
        function_matches = re.findall(function_pattern, content)
        entities.update(function_matches)
        
        # Extract module names
        module_pattern = r'\btorch\.(\w+)\b'
        module_matches = re.findall(module_pattern, content)
        entities.update([f"torch.{match}" for match in module_matches])
        
        # Update metadata
        metadata["topics"] = topics
        metadata["entities"] = entities
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old conversation sessions."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        
        for session_id, metadata in self.conversation_metadata.items():
            last_activity = metadata.get("last_activity", 0)
            if current_time - last_activity > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        # Remove old sessions
        for session_id in sessions_to_remove:
            self.clear_conversation(session_id)
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        return len(sessions_to_remove)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        
        for session_id in self.conversations.keys():
            session_info = self.get_session_info(session_id)
            sessions.append(session_info)
        
        return sessions


class ConversationSummarizer:
    """Summarizes long conversations to maintain context within token limits."""
    
    def __init__(self, max_context_tokens: int = 2000):
        self.max_context_tokens = max_context_tokens
        
        # Simple token estimation (1 token ≈ 4 characters)
        self.chars_per_token = 4
        
        logger.info("Conversation summarizer initialized")
    
    def should_summarize(self, conversation_context: str) -> bool:
        """Check if conversation should be summarized."""
        estimated_tokens = len(conversation_context) / self.chars_per_token
        return estimated_tokens > self.max_context_tokens
    
    def summarize_conversation(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Create a summary of the conversation."""
        if not conversation_history:
            return ""
        
        # Extract key information
        user_messages = [msg.get("content", "") for msg in conversation_history if msg.get("type") == "user"]
        assistant_messages = [msg.get("content", "") for msg in conversation_history if msg.get("type") == "assistant"]
        
        # Create summary
        summary_parts = []
        
        if user_messages:
            summary_parts.append(f"User asked {len(user_messages)} questions:")
            for i, msg in enumerate(user_messages[-3:], 1):  # Last 3 questions
                summary_parts.append(f"{i}. {msg[:100]}...")
        
        if assistant_messages:
            summary_parts.append(f"Assistant provided {len(assistant_messages)} responses")
        
        return "\n".join(summary_parts)
    
    def get_context_with_summary(self, conversation_history: List[Dict[str, Any]], 
                                max_length: Optional[int] = None) -> Tuple[str, bool]:
        """Get conversation context with summary if needed."""
        # Build full context
        context = self._build_full_context(conversation_history)
        
        # Check if summarization is needed
        if self.should_summarize(context):
            # Create summary of older messages
            older_messages = conversation_history[:-3] if len(conversation_history) > 3 else []
            recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            summary = self.summarize_conversation(older_messages)
            recent_context = self._build_full_context(recent_messages)
            
            combined_context = f"Previous conversation summary:\n{summary}\n\nRecent conversation:\n{recent_context}"
            return combined_context, True
        
        return context, False
    
    def _build_full_context(self, messages: List[Dict[str, Any]]) -> str:
        """Build full context from messages."""
        context_parts = []
        
        for message in messages:
            if message.get("type") == "user":
                context_parts.append(f"User: {message.get('content', '')}")
            elif message.get("type") == "assistant":
                context_parts.append(f"Assistant: {message.get('content', '')}")
        
        return "\n".join(context_parts)


# Global conversation memory instance
_conversation_memory = None


def get_conversation_memory() -> ConversationMemory:
    """Get the global conversation memory instance."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory


def main():
    """Test the conversation memory."""
    memory = ConversationMemory()
    summarizer = ConversationSummarizer()
    
    # Test conversation
    session_id = "test_session_123"
    
    # Add some messages
    memory.add_message(session_id, {
        "type": "user",
        "content": "How do I create a tensor in PyTorch?"
    })
    
    memory.add_message(session_id, {
        "type": "assistant",
        "content": "You can create a tensor using torch.tensor(data). Here's an example..."
    })
    
    memory.add_message(session_id, {
        "type": "user",
        "content": "What about torch.nn.Linear?"
    })
    
    # Get conversation history
    history = memory.get_conversation_history(session_id)
    print(f"Conversation history: {len(history)} messages")
    
    # Get context
    context = memory.get_conversation_context(session_id)
    print(f"Context length: {len(context)} characters")
    
    # Get summary
    summary = memory.summarize_conversation(session_id)
    print(f"Conversation summary: {summary}")
    
    # Test summarization
    should_summarize = summarizer.should_summarize(context)
    print(f"Should summarize: {should_summarize}")


if __name__ == "__main__":
    main()
