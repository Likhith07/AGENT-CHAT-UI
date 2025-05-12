import json
import time
from .app_setup import logger # Assuming logger is in app_setup.py
from .state_management import sessions # Assuming sessions is in state_management.py

def _generate_sse_events(thread_id, run_id):
    """Helper function to generate Server-Sent Events for a thread and run.
       Yields ONLY the standard OpenAI Assistants API events.
    """
    logger.info(f"SSE Generation started for thread {thread_id}, run {run_id} (Standard Events Only)")
    
    # 1. Yield thread.run.created (or in_progress)
    run_created_event = {
        # Using run_id for event ID for simplicity, ensure uniqueness if needed
        "id": f"evt_{run_id}_created",
        "object": "thread.run.created",
        "created_at": int(time.time()),
        "data": {
            "id": run_id,
            "object": "thread.run",
            "created_at": int(time.time()),
            "assistant_id": "agent",
            "thread_id": thread_id,
            "status": "queued", # Or "in_progress" if starting immediately
            # Add other relevant run fields if available
        }
    }
    logger.info(f"Yielding {run_created_event['object']} event for run {run_id} with thread_id: {thread_id}")
    yield f"event: {run_created_event['object']}\ndata: {json.dumps(run_created_event['data'])}\n\n"
    time.sleep(0.01)

    # Emit metadata event with run_id for LangGraph SDK
    metadata_event = {
        "run_id": run_id
    }
    yield f"event: metadata\ndata: {json.dumps(metadata_event)}\n\n"
    time.sleep(0.01)
    
    run_inprogress_event = {
        "id": f"evt_{run_id}_inprogress",
        "object": "thread.run.in_progress",
        "created_at": int(time.time()),
        "data": {
            "id": run_id,
            "object": "thread.run",
            "created_at": run_created_event['data']["created_at"],
            "assistant_id": "agent",
            "thread_id": thread_id,
            "status": "in_progress",
            "started_at": int(time.time())
        }
    }
    logger.info(f"Yielding {run_inprogress_event['object']} event for run {run_id}")
    yield f"event: {run_inprogress_event['object']}\ndata: {json.dumps(run_inprogress_event['data'])}\n\n"
    time.sleep(0.01)

    unique_ai_messages = []
    if thread_id in sessions and sessions[thread_id]["messages"]:
        latest_user_msg_index = -1
        for i, msg in enumerate(sessions[thread_id]["messages"]):
            if msg.get("type") == "human":
                latest_user_msg_index = i
        
        ai_messages_to_stream = []
        if latest_user_msg_index != -1:
            ai_messages_to_stream = [msg for i_msg, msg in enumerate(sessions[thread_id]["messages"])
                                    if msg.get("type") == "ai" and i_msg > latest_user_msg_index]
        elif not any(msg.get("type") == "human" for msg in sessions[thread_id]["messages"]):
            ai_messages_to_stream = [msg for msg in sessions[thread_id]["messages"] if msg.get("type") == "ai"]

        seen_contents = set()
        for msg in ai_messages_to_stream:
            content = msg.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_ai_messages.append(msg)
        
    logger.info(f"Found {len(unique_ai_messages)} unique AI messages to stream for run {run_id}")

    # 2. Stream thread.message events for each AI message
    message_creation_timestamp = int(time.time()) # Use consistent timestamp for parts of same message
    for i, ai_message in enumerate(unique_ai_messages):
        message_id = ai_message.get("id", f"msg_{run_id}_{i}") # Generate message ID if missing
        if message_id.startswith("do-not-render-"): continue # Skip internal markers
        
        msg_content = ai_message.get("content", "")
        logger.info(f"Streaming message {i+1}/{len(unique_ai_messages)}: ID={message_id}, Content='{msg_content[:50]}...'" )

        # 2a. Send thread.message.created
        message_created_event_data = {
            "id": message_id,
            "object": "thread.message",
            "created_at": message_creation_timestamp,
            "thread_id": thread_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [], # Content comes in delta
            "assistant_id": "agent",
            "run_id": run_id,
            "attachments": [],
            "metadata": {}
        }
        message_created_event = {
            "id": f"evt_mcreated_{message_id}",
            "object": "thread.message.created",
            "created_at": message_creation_timestamp,
            "data": message_created_event_data
        }
        logger.info(f"Yielding {message_created_event['object']} for msg {message_id}")
        yield f"event: {message_created_event['object']}\ndata: {json.dumps(message_created_event['data'])}\n\n"
        time.sleep(0.01)

        # 2b. Send thread.message.delta (can be broken into chunks if needed)
        # For simplicity, sending the whole message content in one delta
        message_delta_event_data_delta = {
            "content": [
                {
                    "index": 0,
                    "type": "text",
                    "text": {"value": msg_content, "annotations": []}
                }
            ]
        }
        message_delta_event_data = {
            "id": message_id,
            "object": "thread.message.delta",
            "delta": message_delta_event_data_delta
        }
        message_delta_event = {
            "id": f"evt_mdelta_{message_id}",
            "object": "thread.message.delta",
            "created_at": int(time.time()), # Delta timestamp
            "data": message_delta_event_data
        }
        logger.info(f"Yielding {message_delta_event['object']} for msg {message_id}")
        yield f"event: {message_delta_event['object']}\ndata: {json.dumps(message_delta_event['data'])}\n\n"
        time.sleep(0.01)

        # 2c. Send thread.message.completed
        message_completed_event_data = {
            "id": message_id,
            "object": "thread.message",
            "created_at": message_creation_timestamp,
            "thread_id": thread_id,
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": {"value": msg_content, "annotations": []}
            }],
            "assistant_id": "agent",
            "run_id": run_id,
            "attachments": [],
            "metadata": {}
        }
        message_completed_event = {
            "id": f"evt_mcompleted_{message_id}",
            "object": "thread.message.completed",
            "created_at": int(time.time()), # Completion timestamp
            "data": message_completed_event_data
        }
        logger.info(f"Yielding {message_completed_event['object']} for msg {message_id}")
        yield f"event: {message_completed_event['object']}\ndata: {json.dumps(message_completed_event['data'])}\n\n"
        time.sleep(0.01)

    # Emit LangGraph SDK compatible "values" event so React SDK can update UI
    try:
        current_session_messages = sessions.get(thread_id, {}).get("messages", [])
        # Ensure the state snapshot for the 'values' event data strictly matches frontend StateType
        current_state_snapshot = {
            "messages": current_session_messages
            # If StateType had a 'ui' field, it would be included here too.
            # Do NOT include 'thread_id' here; SDK infers it from context or other events.
        }
        logger.info(f"Yielding 'values' event with state data: {json.dumps(current_state_snapshot)}")
        yield f"event: values\ndata: {json.dumps(current_state_snapshot)}\n\n"
    except Exception as e:
        logger.error(f"Error emitting values event for run {run_id}: {e}")

    # Emit a generic 'end' event for the LangGraph SDK before OpenAI's run.completed
    # This might help the SDK finalize the state from 'values' when streamMode: ["values"] is used.
    try:
        logger.info(f"Yielding generic 'end' event for run {run_id}")
        yield "event: end\ndata: Done\n\n"
        time.sleep(0.005)
    except Exception as e:
        logger.error(f"Error emitting generic 'end' event for run {run_id}: {e}")

    # 3. Finally, yield thread.run.completed
    run_completed_event_data = {
        "id": run_id,
        "object": "thread.run",
        "created_at": run_created_event['data']["created_at"],
        "assistant_id": "agent",
        "thread_id": thread_id,
        "status": "completed",
        "started_at": run_inprogress_event['data']["started_at"],
        "completed_at": int(time.time()),
        "expires_at": None,
        "required_action": None,
        "last_error": None,
        "model": "gpt-4o", # Example model
        "instructions": None, # Example
        "tools": [], # Example
        "metadata": {},
        "usage": None # Example usage if available
    }
    run_completed_event = {
        "id": f"evt_{run_id}_completed",
        "object": "thread.run.completed",
        "created_at": int(time.time()),
        "data": run_completed_event_data
    }
    logger.info(f"Yielding {run_completed_event['object']} event for run {run_id}")
    yield f"event: {run_completed_event['object']}\ndata: {json.dumps(run_completed_event['data'])}\n\n"
    
    logger.info(f"SSE Generation completed for run {run_id}") 