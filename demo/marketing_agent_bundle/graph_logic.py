from typing import Literal
import logging
from langgraph.graph import StateGraph, END

from .agent_state import MarketingPlanState # Assuming agent_state.py is in the same directory
# Import node functions from graph_nodes.py
from .graph_nodes import (
    initialize_state,
    extract_business_data,
    gather_competitor_data,
    analyze_marketing_channels,
    refine_marketing_plan,
    generate_final_plan,
    handle_plan_delivery
)

logger = logging.getLogger(__name__)

def should_end(state: MarketingPlanState) -> Literal["continue", END]:
  """Determine if the workflow should end."""
  logger.info(f"🔄 LangGraph Flow Control: should_end - Current stage: {state.get('current_stage', 'unknown')}")

  # Explicit check for final stage
  if state.get("current_stage") == "final":
    logger.info("🔄 LangGraph Flow Control: should_end - Final stage detected, ending workflow")
    return END

  # Detect loops - check for repeated messages of the same content
  # Count occurrences of system-4 messages asking the same question
  focus_questions = [msg for msg in state.get("messages", [])
                    if msg.get("id") == "system-4" and "focus more on social media or search ads" in msg.get("content", "")]

  # LOWER THRESHOLD: More aggressive loop detection - end after just 2 repeated questions instead of 3
  if len(focus_questions) > 2:
    logger.info(f"🔄 LangGraph Flow Control: should_end - Detected {len(focus_questions)} focus questions, forcing workflow end")
    # Set a default budget if needed
    if not state.get("user_input", {}).get("budget"):
      state["user_input"]["budget"] = "$5000"
    # Set a default focus if needed
    if not state.get("user_input", {}).get("focus"):
      state["user_input"]["focus"] = "social media"
    # Update stage to final
    state["current_stage"] = "final"
    return END

  budget_questions = [msg for msg in state.get("messages", [])
                      if msg.get("type") == "ai" and "budget" in msg.get("content", "").lower()]

  # LOWER THRESHOLD: Only allow 2 budget questions before forcing end
  if len(budget_questions) > 2:
    logger.info(f"🔄 LangGraph Flow Control: should_end - Detected {len(budget_questions)} budget questions, forcing workflow end")
    # Set a default budget if needed
    if not state.get("user_input", {}).get("budget"):
      state["user_input"]["budget"] = "$5000"
    # Set a default focus if needed
    if not state.get("user_input", {}).get("focus"):
      state["user_input"]["focus"] = "social media"
    # Update stage to final
    state["current_stage"] = "final"
    return END

  # Detect campaign start questions loop
  campaign_questions = [msg for msg in state.get("messages", [])
                        if msg.get("type") == "ai" and "start the marketing campaign" in msg.get("content", "")]

  # LOWER THRESHOLD: Only allow 2 campaign start questions
  if len(campaign_questions) > 2:
    logger.info("🔄 LangGraph Flow Control: should_end - Detected conversation loop, ending workflow")
    # Update stage to final to ensure proper termination
    state["current_stage"] = "final"
    return END

  # Check if we have a message requesting the final plan
  for message in state.get("messages", []):
    if message.get("type") == "human" and message.get("content"):
      content = message.get("content", "").lower()
      if any(keyword in content for keyword in ["final plan", "generate plan", "create plan", "looks good", "satisfied"]):
        logger.info("🔄 LangGraph Flow Control: should_end - User requested final plan, ending workflow")
        state["current_stage"] = "final"
        return END

  # If we have enough data to generate a plan and user confirms, end the workflow
  has_budget = bool(state.get("user_input", {}).get("budget"))
  has_focus = bool(state.get("user_input", {}).get("focus"))

  if has_budget and has_focus and state.get("current_stage") == "refinement":
    # Look for confirmation words in recent messages
    for message in list(reversed(state.get("messages", [])))[:5]: # Check last 5 messages
      if message.get("type") == "human":
        content = message.get("content", "").lower()
        if any(word in content for word in ["yes", "yeah", "sure", "ok", "okay", "great"]):
          logger.info("🔄 LangGraph Flow Control: should_end - User confirmed, ending workflow")
          state["current_stage"] = "final"
          return END

  logger.info("🔄 LangGraph Flow Control: should_end - Continuing workflow")
  return "continue"

def route_by_stage(state: MarketingPlanState) -> str:
  """Route to the next node based on the current stage."""
  current_stage = state.get("current_stage", "initial")
  logger.info(f"🔄 LangGraph Flow Control: route_by_stage - Routing from stage: {current_stage}")

  # Check for repeated questions that indicate loops
  focus_questions = [msg for msg in state.get("messages", [])
                    if msg.get("id") == "system-4" and "focus more on social media or search ads" in msg.get("content", "")]

  budget_questions = [msg for msg in state.get("messages", [])
                      if msg.get("type") == "ai" and "budget" in msg.get("content", "").lower()]

  campaign_questions = [msg for msg in state.get("messages", [])
                        if msg.get("type") == "ai" and "start the marketing campaign" in msg.get("content", "")]

  # If we're stuck asking the same question, force progression
  if (len(focus_questions) > 3 and current_stage == "analysis") or len(budget_questions) > 3:
    logger.info("🔄 LangGraph Flow Control: route_by_stage - Detected question loop, forcing progression")
    # Set default values if needed
    if not state.get("user_input", {}).get("budget"):
      state["user_input"]["budget"] = "$5000"

    # Force move to refinement
    if current_stage == "analysis":
      logger.info("🔄 LangGraph Flow Control: route_by_stage - Forcing move from analysis to refinement")
      state["current_stage"] = "refinement"
      return "refinement"

  # If we're asking too many times about campaign start, move to final
  if len(campaign_questions) > 3 and current_stage == "refinement":
    logger.info("🔄 LangGraph Flow Control: route_by_stage - Detecting campaign question loop, forcing move to final stage")
    state["current_stage"] = "final"
    return "final"

  return current_stage

# Define the state graph
def build_graph():
  workflow = StateGraph(MarketingPlanState)

  # Add nodes
  workflow.add_node("initial", initialize_state)
  workflow.add_node("data_gathering", extract_business_data)
  workflow.add_node("competitor_analysis", gather_competitor_data)
  workflow.add_node("analysis", analyze_marketing_channels)
  workflow.add_node("refinement", refine_marketing_plan)
  workflow.add_node("final", generate_final_plan)
  workflow.add_node("delivery", handle_plan_delivery) # Add new delivery node

  # Add conditional edges
  workflow.add_conditional_edges(
    "initial",
    should_end,
    {
      "continue": "data_gathering",
      END: END
    }
  )

  workflow.add_conditional_edges(
    "data_gathering",
    should_end,
    {
      "continue": "competitor_analysis",
      END: END
    }
  )

  workflow.add_conditional_edges(
    "competitor_analysis",
    should_end,
    {
      "continue": "analysis",
      END: END
    }
  )

  workflow.add_conditional_edges(
    "analysis",
    should_end,
    {
      "continue": "refinement",
      END: END
    }
  )

  workflow.add_conditional_edges(
    "refinement",
    should_end,
    {
      "continue": "refinement",
      END: "final" # When should_end returns END, go to final stage
    }
  )

  workflow.add_conditional_edges(
    "final",
    lambda x: "continue" if "download" in x.get("messages", [])[-1].get("content", "").lower() or "email" in x.get("messages", [])[-1].get("content", "").lower() else END,
    {
      "continue": "delivery",
      END: END
    }
  )

  workflow.add_conditional_edges(
    "delivery",
    lambda x: END, # Always end after delivery stage
    {
      END: END
    }
  )

  # Set the entry point
  workflow.set_entry_point("initial")

  logger.info("LangGraph workflow compiled with nodes: initial, data_gathering, competitor_analysis, analysis, refinement, final, delivery")

  # Just use debug=True without recursion_limit since it's not supported in this version
  return workflow.compile(debug=True) 