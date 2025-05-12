from flask import Blueprint, request, jsonify, make_response, Response
import time
import json
import uuid

from .app_setup import logger
from .state_management import sessions, threads, sync_threads_and_sessions
from .streaming_utils import _generate_sse_events

from marketing_agent_bundle.marketing_agent import on_message
from marketing_agent_bundle.graph_nodes import generate_final_plan
# build_graph is not directly used by routes, but by on_message indirectly.
# from marketing_agent_bundle.graph_logic import build_graph

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/api/chat/<session_id>/message', methods=['POST'])
def post_message(session_id):
    """Endpoint to post a message to a specific chat session"""
    data = request.json
    
    # Create a new session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": []
        }
    
    # Handle simpler content format that might come from agent-chat-ui
    if isinstance(data, dict) and "content" in data and isinstance(data["content"], str):
        data = {"content": data["content"], "id": str(uuid.uuid4())}
    
    # Process the message
    try:
        result = on_message(sessions[session_id], data)
        sessions[session_id] = result  # Update the session with the result
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        
        # If error occurs, add a fallback message
        if "current_stage" not in sessions[session_id]:
            sessions[session_id]["current_stage"] = "initial"
        
        # Set default values if needed
        if "user_input" not in sessions[session_id]:
            sessions[session_id]["user_input"] = {}
        
        # Add necessary defaults to avoid further problems
        sessions[session_id]["user_input"]["budget"] = "$5000"
        sessions[session_id]["user_input"]["focus"] = "social media"
        sessions[session_id]["user_input"]["start_date"] = "next month"
        
        # Add a message explaining what happened
        sessions[session_id]["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "I noticed we were having trouble proceeding. I'll generate a marketing plan with default settings based on your business profile."
        })
        
        # Generate the final plan
        try:
            result = generate_final_plan(sessions[session_id])
            sessions[session_id] = result
        except Exception as final_error:
            logger.error(f"Error generating final plan: {str(final_error)}")
            sessions[session_id]["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": "I encountered an error while generating your marketing plan. Please try again."
            })
    
    # Return only the messages for agent-chat-ui compatibility
    return jsonify({"messages": sessions[session_id]["messages"]})

@main_routes.route('/api/chat/<session_id>', methods=['GET'])
def get_session(session_id):
    """Endpoint to get all messages in a session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": []
        }
    
    # Return only the messages for agent-chat-ui compatibility
    return jsonify({"messages": sessions[session_id]["messages"]})

@main_routes.route('/api/chat', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": []
    }
    
    # Initialize with a welcome message
    welcome_message = {
        "id": "welcome",
        "type": "ai",
        "content": "Welcome to the AI-Powered Marketing Media Plan Generator! Please provide your business website URL to start."
    }
    
    sessions[session_id]["messages"].append(welcome_message)
    
    return jsonify({
        "session_id": session_id,
        "messages": sessions[session_id]["messages"]
    })

@main_routes.route('/api/graph/agent', methods=['GET'])
def get_graph_info():
    """Return information about the agent graph - this is needed for agent-chat-ui"""
    return jsonify({
        "id": "agent",
        "name": "Marketing Media Plan Generator",
        "description": "This agent helps create a comprehensive marketing media plan based on your business website and preferences."
    })

@main_routes.route('/info', methods=['GET'])
def get_info():
    """Endpoint for agent-chat-ui compatibility, focusing on Assistants API mode."""
    return jsonify({
        "name": "Marketing Media Plan Generator",
        "description": "This agent helps create a comprehensive marketing media plan.",
        "capabilities": {
            "assistants_api_compatible": True,
            "uses_threads": True,
            "streaming_runs": True
        },
        "endpoints": {
            "agents_list": "/api/agents",
            "agent_detail": "/api/agents/{agent_id}",
            "threads": "/threads",
            "threads_search": "/threads/search",
            "messages": "/threads/{thread_id}/messages",
            "runs": "/threads/{thread_id}/runs",
            "stream": "/runs/stream"
        },
        "models": ["gpt-4o"],
        "version": "1.0.0"
    })

# New endpoints for OpenAI Assistants API compatibility
@main_routes.route('/threads', methods=['POST', 'OPTIONS'])
def create_thread():
    """Create a new thread (OpenAI Assistants API compatible)"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-Beta")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    logger.info("Creating new thread via POST /threads")
    thread_id = str(uuid.uuid4())
    
    # Initialize the session for this thread with a welcome message
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
    
    # Add a welcome message (important for first-time visibility)
    welcome_id = str(uuid.uuid4())
    welcome_message = {
        "id": welcome_id,
        "type": "ai",
        "content": "Welcome to the AI-Powered Marketing Media Plan Generator! Please provide your business website URL to start."
    }
    sessions[thread_id]["messages"].append(welcome_message)
    logger.info(f"Added welcome message with ID {welcome_id} to new thread {thread_id}")
    
    # This is the standard OpenAI Thread object response
    thread_response_data = {
        "id": thread_id,
        "object": "thread",
        "created_at": int(time.time()),
        "metadata": {}
    }
    
    # Store a reference to the thread object itself
    threads[thread_id] = thread_response_data

    logger.info(f"Created thread with ID: {thread_id}")
    logger.info(f"Responding to POST /threads with Thread object: {json.dumps(thread_response_data)}")
    
    # Ensure threads and sessions are kept in sync
    sync_threads_and_sessions()
    
    response = jsonify(thread_response_data)
    response.headers.add('Content-Type', 'application/json')
    return response

@main_routes.route('/threads/<thread_id>/messages', methods=['POST', 'GET', 'OPTIONS'])
def handle_thread_messages(thread_id):
    """Handle thread messages (OpenAI Assistants API compatible)"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        return response
    
    logger.info(f"Thread messages request for thread {thread_id}, method: {request.method}")
    
    # Ensure threads and sessions are in sync
    sync_threads_and_sessions()
    
    # Create thread if it doesn't exist
    if thread_id not in threads:
        threads[thread_id] = {
            "id": thread_id,
            "object": "thread",
            "created_at": int(time.time()),
            "metadata": {}
        }
    
    # Initialize the sessions data structure for this thread if needed
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
        
        # Add welcome message if this is a new session
        welcome_id = str(uuid.uuid4())
        welcome_message = {
            "id": welcome_id,
            "type": "ai",
            "content": "Welcome to the AI-Powered Marketing Media Plan Generator! Please provide your business website URL to start."
        }
        sessions[thread_id]["messages"].append(welcome_message)
        logger.info(f"Added welcome message with ID {welcome_id} to new thread {thread_id}")
    
    if request.method == 'POST':
        # Add a message to the thread
        data = request.json
        logger.info(f"Received message for thread {thread_id}: {data}")
        
        # Extract content based on OpenAI API format
        message_content = ""
        
        # First try content as array of content parts
        content_parts = data.get('content', [])
        if isinstance(content_parts, list) and len(content_parts) > 0:
            for part in content_parts:
                if part.get('type') == 'text':
                    message_content += part.get('text', {}).get('value', '')
            logger.info(f"Extracted message from content parts array: '{message_content}'")
        
        # If still empty, try as direct content string
        if not message_content:
            if 'content' in data and isinstance(data['content'], str):
                message_content = data['content']
                logger.info(f"Extracted message from direct content string: '{message_content}'")
        
        # If still empty, try text property
        if not message_content and 'text' in data:
            message_content = data.get('text', '')
            logger.info(f"Extracted message from text property: '{message_content}'")
        
        # Last resort - look through all properties for a string
        if not message_content:
            for key, value in data.items():
                if isinstance(value, str) and value:
                    message_content = value
                    logger.info(f"Extracted message from property {key}: '{message_content}'")
                    break
        
        logger.info(f"Final extracted message content: '{message_content}'")
        
        if message_content:
            message_id = str(uuid.uuid4())
            
            # Create message object in OpenAI format
            message = {
                "id": message_id,
                "object": "thread.message",
                "created_at": int(time.time()),
                "thread_id": thread_id,
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": {
                            "value": message_content,
                            "annotations": []
                        }
                    }
                ]
            }
            
            # Check if message is duplicate
            duplicate = False
            for msg in sessions[thread_id]["messages"]:
                if msg.get("type") == "human" and msg.get("content") == message_content:
                    duplicate = True
                    message_id = msg.get("id")  # Use existing ID
                    break
            
            if not duplicate:
                # Add message to our internal format
                user_message = {
                    "id": message_id,
                    "type": "human",
                    "content": message_content
                }
                sessions[thread_id]["messages"].append(user_message)
                logger.info(f"Added user message with ID {message_id} to thread {thread_id}")
                
                # Process with agent
                try:
                    logger.info(f"Processing message with agent for thread {thread_id}")
                    
                    # Generate response
                    result = on_message(sessions[thread_id], {"id": message_id, "content": message_content})
                    
                    # Update session
                    sessions[thread_id] = result
                    logger.info(f"Processing complete. New messages: {len(result.get('messages', []))}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    # Add error message
                    sessions[thread_id]["messages"].append({
                        "id": str(uuid.uuid4()),
                        "type": "ai",
                        "content": "I encountered an error. Let me try to generate a simpler response."
                    })
            else:
                logger.info(f"Duplicate message detected, skipping processing")
            
            # Ensure threads and sessions remain in sync
            sync_threads_and_sessions()
            return jsonify(message)
        else:
            logger.warning(f"Empty message content for thread {thread_id}")
            return jsonify({"error": "Empty message content"}), 400
    
    elif request.method == 'GET':
        # Return all messages in the thread
        messages_data = []
        
        # Convert internal format to OpenAI format
        for msg in sessions[thread_id]["messages"]:
            role = "user" if msg.get("type") == "human" else "assistant"
            content = msg.get("content", "")
            
            messages_data.append({
                "id": msg.get("id", str(uuid.uuid4())),
                "object": "thread.message",
                "created_at": int(time.time()),
                "thread_id": thread_id,
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": {
                            "value": content,
                            "annotations": []
                        }
                    }
                ]
            })
        
        logger.info(f"Returning {len(messages_data)} messages for thread {thread_id}")
        return jsonify({
            "object": "list",
            "data": messages_data
        })

@main_routes.route('/threads/<thread_id>/runs', methods=['POST', 'GET', 'OPTIONS'])
def handle_thread_runs(thread_id):
    """Handle non-streaming runs. POST creates a run, GET lists runs."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-Beta, X-Thread-ID")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    logger.info(f"Thread runs request for thread {thread_id}, method: {request.method}")
    sync_threads_and_sessions() # Ensure state consistency

    if thread_id not in sessions:
        logger.error(f"Thread {thread_id} not found in sessions for handle_thread_runs.")
        return jsonify({"error": f"Thread {thread_id} not found"}), 404

    if request.method == 'POST':
        data = request.json or {}
        assistant_id = data.get('assistant_id', 'agent')
        logger.info(f"POST /threads/{thread_id}/runs (non-streaming) received data: {data}")

        if data.get('stream') == True:
            logger.error(f"Streaming request sent to non-streaming endpoint /threads/{thread_id}/runs")
            return jsonify({"error": "Streaming not supported at this endpoint. Use a designated streaming endpoint."}), 400

        run_id = str(uuid.uuid4())
        logger.info(f"Creating non-streaming run {run_id} for thread {thread_id}.")

        # --- Refined Message Extraction for Non-Streaming Run ---
        new_user_message_content = None
        input_for_on_message = None

        if "additional_messages" in data and isinstance(data["additional_messages"], list) and len(data["additional_messages"]) > 0:
            last_additional_message = data["additional_messages"][-1]
            if last_additional_message.get("role") == "user":
                content_parts = last_additional_message.get("content", [])
                if isinstance(content_parts, str):
                    new_user_message_content = content_parts
                elif isinstance(content_parts, list):
                    text_val = ""
                    for part in content_parts:
                        if part.get("type") == "text":
                            text_val += part.get("text", {}).get("value", "")
                    new_user_message_content = text_val
                if new_user_message_content:
                    input_for_on_message = {"id": last_additional_message.get("id", "msg-" + run_id), "content": new_user_message_content}
                    logger.info(f"Extracted message from additional_messages for run {run_id}")

        elif "instructions" in data and data["instructions"] and not new_user_message_content:
            # Treat instructions as the user message if no user message found in additional_messages
            new_user_message_content = data["instructions"]
            input_for_on_message = {"id": "instr-" + run_id, "content": new_user_message_content}
            logger.info(f"Extracted message from instructions for run {run_id}")
        # --- End Refined Message Extraction ---

        if new_user_message_content and input_for_on_message:
            # Add user message to session state if needed (on_message might handle this too)
            is_duplicate = False
            for msg in sessions[thread_id].get('messages', []):
                 if msg.get("id") == input_for_on_message["id"] or \
                   (msg.get("type") == "human" and msg.get("content") == new_user_message_content):
                     is_duplicate = True; break
            if not is_duplicate:
                 sessions[thread_id].setdefault("messages", []).append({"id": input_for_on_message["id"], "type": "human", "content": new_user_message_content})
            
            try:
                # Call your agent/graph logic
                sessions[thread_id] = on_message(sessions[thread_id], input_for_on_message)
            except Exception as e:
                logger.error(f"Error in on_message for non-streaming run {run_id}: {str(e)}")
                # TODO: Decide how to handle errors in non-streaming runs (e.g., update run status)
        elif sessions[thread_id].get("messages") and sessions[thread_id]["messages"][-1]["type"] == "human":
             # If no new message, but last was human, run based on last message
            try:
                last_human_message = sessions[thread_id]["messages"][-1]
                sessions[thread_id] = on_message(sessions[thread_id], last_human_message)
            except Exception as e:
                logger.error(f"Error in on_message (continuation) for non-streaming run {run_id}: {str(e)}")

        # Return the Run object (immediately completed for simplicity here)
        non_stream_run = {
            "id": run_id,
            "object": "thread.run",
            "created_at": int(time.time()),
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "completed",
            "required_action": None,
            "last_error": None,
            "expires_at": None,
            "started_at": int(time.time()),
            "completed_at": int(time.time()),
            "model": "gpt-4o",
            "instructions": data.get("instructions"),
            "tools": [],
            "metadata": {}
        }
        return jsonify(non_stream_run)

    elif request.method == 'GET':
        logger.info(f"GET /threads/{thread_id}/runs - Listing runs.")
        # Return empty list as run storage isn't implemented
        return jsonify({"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False})

# NEW Route for thread-specific streaming runs
@main_routes.route('/threads/<thread_id>/runs/stream', methods=['POST', 'OPTIONS'])
def handle_thread_specific_stream(thread_id):
    """Handle streaming runs for a specific thread. Expected by SDK when threadId is known."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-Beta, X-Thread-ID")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    data = request.json or {}
    logger.info(f"POST /threads/{thread_id}/runs/stream received data: {data}")
    sync_threads_and_sessions() # Ensure state consistency

    if thread_id not in sessions:
        logger.warning(f"Thread {thread_id} (from path) not found in sessions. Initializing.")
        sessions[thread_id] = {
            "messages": [], "business_info": {}, "competitor_info": [], "marketing_channels": [],
            "budget_allocation": {}, "ad_creatives": [], "user_input": {}, "current_stage": "initial"
        }
        welcome_id = str(uuid.uuid4())
        welcome_message = { "id": welcome_id, "type": "ai", "content": "Welcome! Please provide your business website URL."}
        sessions[thread_id]["messages"].append(welcome_message)
        logger.info(f"Initialized session and added welcome message for thread {thread_id}")

    # --- Message Extraction Logic (Prioritize instructions, then additional_messages) ---
    new_user_message_content = None
    input_for_on_message = None
    message_source = "Unknown"

    if "instructions" in data and data["instructions"]:
        new_user_message_content = data["instructions"]
        input_for_on_message = {"id": "instr-" + str(uuid.uuid4()), "content": new_user_message_content}
        message_source = "instructions"
    elif "additional_messages" in data and isinstance(data["additional_messages"], list) and len(data["additional_messages"]) > 0:
        last_additional_message = data["additional_messages"][-1]
        if last_additional_message.get("role") == "user":
            content_parts = last_additional_message.get("content", [])
            temp_content = ""
            if isinstance(content_parts, str):
                temp_content = content_parts
            elif isinstance(content_parts, list):
                for part in content_parts:
                    if part.get("type") == "text":
                        temp_content += part.get("text", {}).get("value", "")
            if temp_content:
                new_user_message_content = temp_content
                input_for_on_message = {"id": last_additional_message.get("id", "msg-" + str(uuid.uuid4())), "content": new_user_message_content}
                message_source = "additional_messages"
    elif 'input' in data and isinstance(data['input'], dict):
        input_data_val = data['input']
        if 'messages' in input_data_val and isinstance(input_data_val['messages'], list) and input_data_val['messages']:
            latest_message_from_input = input_data_val['messages'][-1]
            if isinstance(latest_message_from_input, dict):
                msg_content = latest_message_from_input.get('content')
                temp_content = ""
                if isinstance(msg_content, str): temp_content = msg_content
                elif isinstance(msg_content, list):
                    for part in msg_content:
                        if part.get("type") == "text": temp_content += part.get("text", {}).get("value", "")
                if temp_content:
                    new_user_message_content = temp_content
                    input_for_on_message = {"id": latest_message_from_input.get('id', str(uuid.uuid4())), "content": new_user_message_content}
                    message_source = "input.messages"
    # --- End Message Extraction ---
    
    run_id = f"run_{str(uuid.uuid4())[:8]}"

    if new_user_message_content and input_for_on_message:
        logger.info(f"Processing new message (from {message_source}) for thread {thread_id} via specific stream: '{new_user_message_content[:50]}...'")
        is_duplicate = False
        for msg in sessions[thread_id].get('messages', []):
             if msg.get("id") == input_for_on_message["id"] or \
               (msg.get("type") == "human" and msg.get("content") == new_user_message_content):
                 is_duplicate = True; break
        if not is_duplicate:
             sessions[thread_id].setdefault("messages", []).append({"id": input_for_on_message["id"], "type": "human", "content": new_user_message_content})
        try:
            sessions[thread_id] = on_message(sessions[thread_id], input_for_on_message)
        except Exception as e:
            logger.error(f"Error in on_message from /threads/{thread_id}/runs/stream: {str(e)}")
            sessions[thread_id].setdefault("messages", []).append({"id": "err-" + run_id, "type": "ai", "content": "Error processing message."})
    elif sessions[thread_id].get("messages") and sessions[thread_id]["messages"][-1]["type"] == "human":
        last_human_message = sessions[thread_id]["messages"][-1]
        logger.info(f"No new message in stream request, continuing from last human message for thread {thread_id}: '{last_human_message.get('content', '')[:50]}...'")
        try:
            sessions[thread_id] = on_message(sessions[thread_id], last_human_message)
        except Exception as e:
            logger.error(f"Error in on_message (continuation) from /threads/{thread_id}/runs/stream: {str(e)}")
            sessions[thread_id].setdefault("messages", []).append({"id": "err-" + run_id, "type": "ai", "content": "Error in continuation."})
    else:
        logger.info(f"No new message and last not human for thread {thread_id} in specific stream. Streaming existing AI messages or welcome.")
        if not sessions[thread_id].get("messages"):
             welcome_id = str(uuid.uuid4())
             welcome_message = { "id": welcome_id, "type": "ai", "content": "Welcome! How can I assist with your marketing plan today?"}
             sessions[thread_id]["messages"].append(welcome_message)

    # Generate and stream the SSE events using the resolved thread_id
    response = Response(_generate_sse_events(thread_id, run_id), mimetype="text/event-stream")
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')
    response.headers.add('Transfer-Encoding', 'chunked')
    return response

# NEW Route for fetching thread history
@main_routes.route('/threads/<thread_id>/history', methods=['GET', 'POST', 'OPTIONS'])
def get_thread_history(thread_id):
    """Return message history array for a specific thread. Accepts GET and POST."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-Beta")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    logger.info(f"{request.method} /threads/{thread_id}/history request (fetching message array)")
    # sync_threads_and_sessions() # REMOVED: Avoid potential blocking during request

    # Directly check sessions dictionary
    if thread_id not in sessions or not sessions[thread_id].get("messages"):
        logger.warning(f"No messages found or thread {thread_id} does not exist for history request.")
        return jsonify([])

    # Access state directly - assumes sync happens elsewhere reliably
    thread_session = sessions.get(thread_id, {})
    current_messages = thread_session.get("messages", [])

    messages_data = []
    for msg in current_messages:
        role = "user" if msg.get("type") == "human" else "assistant"
        content_value = msg.get("content", "")
        msg_id = msg.get("id", str(uuid.uuid4()))
        if msg_id.startswith("do-not-render-"):
            msg_id = msg_id[len("do-not-render-"):]
            if not msg_id: continue
        
        created_at = msg.get("created_at", int(time.time()))

        message_obj = {
            "id": msg_id,
            "object": "thread.message",
            "created_at": created_at,
            "thread_id": thread_id,
            "status": "completed",
            "role": role,
            "content": [{
                "type": "text",
                "text": {"value": content_value, "annotations": []}
            }],
            "assistant_id": "agent" if role == "assistant" else None,
            "run_id": None,
            "attachments": [],
            "metadata": {},
        }
        messages_data.append(message_obj)

    logger.info(
        f"Returning {len(messages_data)} messages as Assistants API style list object for thread {thread_id} history"
    )
    # Return using standard Assistants API format: {"object": "list", "data": [...]
    return jsonify({
        "object": "list",
        "data": messages_data,
    })

@main_routes.route('/runs/stream', methods=['OPTIONS', 'POST'])
def runs_stream_primary():
    """Handle streaming runs started without a thread_id in the path."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-Beta, X-Thread-ID")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    data = request.json or {}
    logger.info(f"POST /runs/stream (primary) received data: {data}")
    sync_threads_and_sessions()

    # --- Thread ID Resolution ---
    thread_id = None
    if 'thread_id' in data:
        thread_id = data.get('thread_id')
        logger.info(f"Found thread_id directly at top level: {thread_id}")
    elif 'thread_id' in data.get('input', {}):
        thread_id = data.get('input').get('thread_id')
        logger.info(f"Found thread_id directly in input: {thread_id}")
    elif 'id' in data:
        thread_id = data.get('id')
        logger.info(f"Found thread_id in 'id' field: {thread_id}")
    elif 'input' in data and isinstance(data['input'], dict):
        input_data_val = data['input']
        if 'thread_id' in input_data_val:
            thread_id = input_data_val.get('thread_id')
            logger.info(f"Found thread_id in input object: {thread_id}")
        elif 'id' in input_data_val:
            thread_id = input_data_val.get('id')
            logger.info(f"Found thread_id in input.id: {thread_id}")
        elif 'messages' in input_data_val and isinstance(input_data_val['messages'], list) and input_data_val['messages']:
            first_message = input_data_val['messages'][0]
            if isinstance(first_message, dict) and 'thread_id' in first_message:
                thread_id = first_message.get('thread_id')
                logger.info(f"Found thread_id in first message: {thread_id}")
    if not thread_id and request.headers.get('X-Thread-ID'):
        thread_id = request.headers.get('X-Thread-ID')
        logger.info(f"Found thread_id in X-Thread-ID header: {thread_id}")
    
    if thread_id and isinstance(thread_id, str) and ' in ' in thread_id:
        original_thread_id = thread_id
        thread_id = thread_id.split(' in ')[0]
        logger.info(f"Cleaned thread_id from '{original_thread_id}' to '{thread_id}'")

    if not thread_id:
        if threads:
            thread_id = list(threads.keys())[-1]
            logger.warning(f"No thread_id found in /runs/stream request, falling back to most recent: {thread_id}")
        else:
            logger.warning("No thread_id in /runs/stream and no existing threads. Creating a new one implicitly.")
            try:
                # Simulate create_thread() which is also on this blueprint
                # This is a bit tricky as we can't call it directly from another route in the same file easily without app context
                # For simplicity, directly implement the core logic of create_thread()
                temp_thread_id = str(uuid.uuid4())
                sessions[temp_thread_id] = {
                    "messages": [], "business_info": {}, "competitor_info": [], "marketing_channels": [],
                    "budget_allocation": {}, "ad_creatives": [], "user_input": {}, "current_stage": "initial"
                }
                welcome_id = str(uuid.uuid4())
                welcome_message = {"id": welcome_id, "type": "ai", "content": "Welcome! Provide business URL."}
                sessions[temp_thread_id]["messages"].append(welcome_message)
                threads[temp_thread_id] = {"id": temp_thread_id, "object": "thread", "created_at": int(time.time()), "metadata": {}}
                thread_id = temp_thread_id
                sync_threads_and_sessions() # Ensure it's fully registered
                logger.info(f"Implicitly created new thread {thread_id} for /runs/stream")
            except Exception as e:
                logger.error(f"Failed to implicitly create thread for /runs/stream: {e}")
                return jsonify({"error": "Failed to initialize chat thread."}), 500
    # --- End Thread ID Resolution ---

    logger.info(f"Processing /runs/stream using resolved thread_id: {thread_id}")
    
    # Ensure session exists (might have been implicitly created)
    if thread_id not in sessions:
        logger.warning(f"Thread {thread_id} (from /runs/stream logic) not found even after potential implicit creation. Initializing.")
        sessions[thread_id] = {
            "messages": [], "business_info": {}, "competitor_info": [], "marketing_channels": [],
            "budget_allocation": {}, "ad_creatives": [], "user_input": {}, "current_stage": "initial"
        }
        welcome_id = str(uuid.uuid4())
        welcome_message = { "id": welcome_id, "type": "ai", "content": "Welcome from /runs/stream! Provide website URL."}
        sessions[thread_id]["messages"].append(welcome_message)
        logger.info(f"Force-initialized session {thread_id} from /runs/stream.")

    # --- Message Extraction (Mirrors thread-specific endpoint) ---
    new_user_message_content = None
    input_for_on_message = None
    message_source = "Unknown"

    if "instructions" in data and data["instructions"]:
        new_user_message_content = data["instructions"]
        input_for_on_message = {"id": "instr-" + str(uuid.uuid4()), "content": new_user_message_content}
        message_source = "instructions"
    elif "additional_messages" in data and isinstance(data["additional_messages"], list) and len(data["additional_messages"]) > 0:
        last_additional_message = data["additional_messages"][-1]
        if last_additional_message.get("role") == "user":
            content_parts = last_additional_message.get("content", [])
            temp_content = ""
            if isinstance(content_parts, str):
                temp_content = content_parts
            elif isinstance(content_parts, list):
                for part in content_parts:
                    if part.get("type") == "text": temp_content += part.get("text", {}).get("value", "")
            if temp_content:
                new_user_message_content = temp_content
                input_for_on_message = {"id": last_additional_message.get("id", "msg-" + str(uuid.uuid4())), "content": new_user_message_content}
                message_source = "additional_messages"
    elif 'input' in data and isinstance(data['input'], dict):
        input_data_val = data['input']
        if 'messages' in input_data_val and isinstance(input_data_val['messages'], list) and input_data_val['messages']:
            latest_message_from_input = input_data_val['messages'][-1]
            if isinstance(latest_message_from_input, dict):
                msg_content = latest_message_from_input.get('content')
                temp_content = ""
                if isinstance(msg_content, str): temp_content = msg_content
                elif isinstance(msg_content, list):
                    for part in msg_content:
                        if part.get("type") == "text": temp_content += part.get("text", {}).get("value", "")
                if temp_content:
                    new_user_message_content = temp_content
                    input_for_on_message = {"id": latest_message_from_input.get('id', str(uuid.uuid4())), "content": new_user_message_content}
                    message_source = "input.messages"
    # --- End Message Extraction ---
    
    run_id = f"run_{str(uuid.uuid4())[:8]}"

    if new_user_message_content and input_for_on_message:
        logger.info(f"Processing new message (from {message_source}) for thread {thread_id} via primary stream: '{new_user_message_content[:50]}...'")
        is_duplicate = False
        for msg in sessions[thread_id].get('messages', []):
             if msg.get("id") == input_for_on_message["id"] or \
               (msg.get("type") == "human" and msg.get("content") == new_user_message_content):
                 is_duplicate = True; break
        if not is_duplicate:
             sessions[thread_id].setdefault("messages", []).append({"id": input_for_on_message["id"], "type": "human", "content": new_user_message_content})
        try:
            sessions[thread_id] = on_message(sessions[thread_id], input_for_on_message)
        except Exception as e:
            logger.error(f"Error in on_message from /runs/stream (primary): {str(e)}")
            sessions[thread_id].setdefault("messages", []).append({"id": "err-" + run_id, "type": "ai", "content": "Error processing message."})
    elif sessions[thread_id].get("messages") and sessions[thread_id]["messages"][-1]["type"] == "human":
        last_human_message = sessions[thread_id]["messages"][-1]
        logger.info(f"No new message in primary stream, continuing from last human message for thread {thread_id}: '{last_human_message.get('content', '')[:50]}...'")
        try:
            sessions[thread_id] = on_message(sessions[thread_id], last_human_message)
        except Exception as e:
            logger.error(f"Error in on_message (continuation) from /runs/stream (primary): {str(e)}")
            sessions[thread_id].setdefault("messages", []).append({"id": "err-" + run_id, "type": "ai", "content": "Error in continuation."})
    else:
        logger.info(f"No new message and last not human for thread {thread_id} in primary stream. Streaming existing AI messages or welcome.")
        if not sessions[thread_id].get("messages"):
             welcome_id = str(uuid.uuid4())
             welcome_message = { "id": welcome_id, "type": "ai", "content": "Welcome! How can I assist?"}
             sessions[thread_id]["messages"].append(welcome_message)

    # Generate and stream the SSE events using the resolved thread_id
    response = Response(_generate_sse_events(thread_id, run_id), mimetype="text/event-stream")
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')
    response.headers.add('Transfer-Encoding', 'chunked')
    return response 