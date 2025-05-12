import time
import uuid
from .app_setup import logger # Assuming logger is in app_setup.py

# In-memory storage for chat sessions
sessions = {}
threads = {}

# Function to ensure threads and sessions are in sync
def sync_threads_and_sessions():
    """Synchronize threads and sessions to ensure both dictionaries have entries for all IDs"""
    # Create thread entries for any existing sessions without a thread
    for session_id in sessions:
        if session_id not in threads:
            threads[session_id] = {
                "id": session_id,
                "object": "thread",
                "created_at": int(time.time()),
                "metadata": {}
            }
            logger.info(f"Created missing thread object for session {session_id}")
    
    # Create empty sessions for any threads without a session
    for thread_id in threads:
        if thread_id not in sessions:
            sessions[thread_id] = {
                "messages": [],
                "business_info": {},
                "competitor_info": [],
                "marketing_channels": [],
                "budget_allocation": {},
                "ad_creatives": [],
                "user_input": {},
                "current_stage": "initial"
            }
            # Add welcome message
            welcome_id = str(uuid.uuid4())
            welcome_message = {
                "id": welcome_id,
                "type": "ai",
                "content": "Welcome to the AI-Powered Marketing Media Plan Generator! Please provide your business website URL to start."
            }
            sessions[thread_id]["messages"].append(welcome_message)
            logger.info(f"Created missing session for thread {thread_id} with welcome message")
    
    logger.info(f"Thread/session sync complete. {len(threads)} threads and {len(sessions)} sessions exist.") 