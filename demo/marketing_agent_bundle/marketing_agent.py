from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
import uuid

# Import modularized components
from .agent_state import MarketingPlanState
# WebsiteAnalysisTool is not directly used here anymore, but it's part of the agent's tools conceptually.
# from .agent_tools import WebsiteAnalysisTool 
from .response_analyzer import analyze_user_response
# Import only the node functions directly called by on_message, others are managed by the graph.
from .graph_nodes import extract_business_data, generate_final_plan 
from .graph_logic import build_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the Tavily API (if used directly by any remaining logic here, otherwise this can be moved too)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Create the agent by building the graph
marketing_agent = build_graph()

# Update the on_message function to use intelligent analysis for each stage
def on_message(state: MarketingPlanState, message: Dict[str, Any]):
  """Process new messages based on current conversation stage"""
  try:
    # Initialize state if needed
    if "messages" not in state:
      state["messages"] = []

    if "business_info" not in state:
      state["business_info"] = {}

    if "competitor_info" not in state:
      state["competitor_info"] = []

    if "marketing_channels" not in state:
      state["marketing_channels"] = []

    if "budget_allocation" not in state:
      state["budget_allocation"] = {}

    if "ad_creatives" not in state:
      state["ad_creatives"] = []

    if "user_input" not in state:
      state["user_input"] = {}

    if "current_stage" not in state:
      state["current_stage"] = "initial"

    # Get the message content and ID
    message_content = message.get("content", "")
    message_id = message.get("id", str(uuid.uuid4()))

    logger.info(f"Processing message (ID: {message_id}) in stage: {state['current_stage']}")
    logger.info(f"Message content: '{message_content[:100]}...' if len(message_content) > 100 else message_content")

    # Check if this message ID already exists to prevent duplicates
    existing_ids = [msg.get("id") for msg in state["messages"]]

    if message_id not in existing_ids:
      # Add the user message to the state
      logger.info(f"Adding user message with ID {message_id} to state")
      state["messages"].append({
        "id": message_id,
        "type": "human",
        "content": message_content
      })

      # For first message with URL, ensure we provide a clear AI response
      if state["current_stage"] == "initial" and "http" in message_content and "." in message_content:
        # Store the URL in user_input
        state["user_input"]["website"] = message_content
        logger.info(f"Stored URL in user_input: {message_content}")
    else:
      logger.info(f"Message with ID {message_id} already exists in state, skipping addition")

    # First check if this is just a greeting message with no URL
    ai_messages = [msg for msg in state.get("messages", []) if msg.get("type") == "ai"]
    last_ai_message = ai_messages[-1] if ai_messages else None
    last_ai_content = last_ai_message.get("content", "").lower() if last_ai_message else ""

    # Get greeting intent using intelligent analysis
    if state["current_stage"] == "initial" and not "http" in message_content:
      llm = ChatOpenAI(model="gpt-4o", temperature=0)
      greeting_prompt = f"""
      Analyze this user message: "{message_content}"
      
      Is this a greeting or introduction without any specific business URL?
      Return "yes" if it's just a greeting, or "no" if it contains substantive information.
      ONLY return "yes" or "no".
      """

      try:
        greeting_result = llm.invoke(greeting_prompt)
        is_greeting = greeting_result.content.strip().lower() == "yes"

        if is_greeting:
          # Respond with a welcome message and ask for the URL
          welcome_response = {
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "Hello! Welcome to the Marketing Media Plan Generator. Please provide your business website URL to analyze (starting with http:// or https://)."
          }
          logger.info(f"Adding welcome response: {welcome_response['id']}")
          state["messages"].append(welcome_response)
          logger.info("Handled greeting message, requesting URL")
          return state
      except:
        # If analysis fails, continue with normal flow
        pass

    # Follow a conversation flow based on the current stage and user messages

    # STEP 1: Extract Business URL and Industry Confirmation
    if state["current_stage"] == "initial" and "http" in message_content:
      # Process the URL using extract_business_data directly
      return extract_business_data(state)

    # STEP 2: Confirm Industry and Move to Budget Question
    if state["current_stage"] == "data_gathering":
      # Use intelligent analysis to determine if the user confirmed their industry
      context_info = {"industry": state.get("business_info", {}).get("industry", "")}
      analysis = analyze_user_response(message_content, context_info, "industry_confirmation")

      if analysis.get("confirmed", False):
        # User confirmed their industry, ask for monthly budget
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "What is your monthly budget for marketing?"
        })
        state["current_stage"] = "analysis" # Move to budget stage
        return state
      elif analysis.get("corrected_industry"):
        # User provided a different industry, update it and confirm
        state["business_info"]["industry"] = analysis.get("corrected_industry")
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": f"Thank you for the correction. So your business is in the {analysis.get('corrected_industry')} industry. What is your monthly budget for marketing?"
        })
        state["current_stage"] = "analysis" # Move to budget stage
        return state
      else:
        # User's response was unclear, ask for clarification
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "I'm not sure if I understood correctly. Could you confirm what industry your business is in?"
        })
        return state

    # STEP 3: Process Budget Information
    if state["current_stage"] == "analysis" and not state.get("user_input", {}).get("budget"):
      # Use intelligent analysis to extract budget information
      context_info = {"industry": state.get("business_info", {}).get("industry", "")}
      analysis = analyze_user_response(message_content, context_info, "budget_extraction")

      if analysis.get("amount"):
        # Successfully extracted a budget amount
        currency_symbol = analysis.get("currency_symbol", "$")

        # Handle Indian currency format specially
        if "crore" in analysis.get("original_format", "").lower() or "lakh" in analysis.get("original_format", "").lower():
          # Use the original format in displayed budget to maintain cultural context
          budget_display = analysis.get("original_format", "")
          # Store the converted standard value for calculations
          budget_value = analysis.get("converted_standard_value", analysis.get("amount", ""))
        else:
          # For standard formats, just use the amount directly
          budget_value = analysis.get("amount", "")
          budget_display = f"{currency_symbol}{budget_value}"

        currency = analysis.get("currency", "dollars")

        # Store the budget information
        state["user_input"]["budget"] = budget_display
        state["user_input"]["budget_value"] = budget_value # Store numerical value for calculations
        state["user_input"]["currency"] = currency
        state["user_input"]["currency_symbol"] = currency_symbol
        logger.info(f"Extracted budget: {budget_display} ({currency})")

        # Generate a personalized response based on the industry and budget
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response_prompt = f"""
        Generate a friendly, conversational response to acknowledge the user's marketing budget and ask about their marketing focus preference.
        
        Budget: {budget_display} {currency} ({analysis.get('period', 'monthly')})
        Industry: {state.get("business_info", {}).get("industry", "")}
        
        Your response should:
        1. Acknowledge the budget accurately (be especially careful with Indian currency formats like crores and lakhs)
        2. Ask if they want to focus more on social media marketing, search ads, or have a balanced approach
        3. Be specific to their industry if possible
        
        Keep your response natural, concise, and friendly.
        """

        try:
          response_result = llm.invoke(response_prompt)
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": response_result.content.strip()
          })
        except:
          # Fallback to a simple response
          budget_message_display = state["user_input"]["budget"]
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": f"Great! I'll plan with a budget of {budget_message_display}. Would you like to focus more on social media or search ads?"
          })
        # After budget, current_stage remains 'analysis' but we expect focus next.
        # The next part of this 'analysis' stage will handle focus.
        return state
      else:
        # Couldn't extract a budget, ask for clarification
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "I couldn't understand the budget amount. Could you please provide your monthly marketing budget? For example, '$5000', '₹50,000', '₹10 lakhs', or '₹2 crores'."
        })
        return state

    # STEP 4: Process Marketing Focus Preference (social media, search ads, or both)
    # This is still part of the 'analysis' stage if budget is set but focus is not.
    if state["current_stage"] == "analysis" and state.get("user_input", {}).get("budget") and not state.get("user_input", {}).get("focus"):
      # Use intelligent analysis to determine marketing focus preference
      context_info = {
        "industry": state.get("business_info", {}).get("industry", ""),
        "budget": state.get("user_input", {}).get("budget", "")
      }
      analysis = analyze_user_response(message_content, context_info, "marketing_focus")

      if analysis.get("confidence", 0) >= 0.5 and analysis.get("primary_focus") in ["social_media", "search_ads", "balanced"]:
        # Map to our internal focus values
        focus_map = {
          "social_media": "social media",
          "search_ads": "search ads",
          "balanced": "balanced"
        }
        user_preference = focus_map[analysis.get("primary_focus")]
        logger.info(f"Determined user focus preference: {user_preference} with confidence {analysis.get('confidence')}")

        # Store the focus preference
        state["user_input"]["focus"] = user_preference

        # Also store any specific platforms or goals mentioned for later use
        if analysis.get("mentioned_platforms"):
          state["user_input"]["mentioned_platforms"] = analysis.get("mentioned_platforms")

        if analysis.get("marketing_goals"):
          state["user_input"]["marketing_goals"] = analysis.get("marketing_goals")

        # Generate a personalized follow-up based on their focus preference
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        if user_preference == "social media":
          # For social media focus, ask about Instagram in a way relevant to their industry
          prompt = f"""
          Generate a personalized question asking if the user would like to allocate a larger portion of their budget to Instagram ads.
          
          Consider these details:
          - Their business is in the {state.get("business_info", {}).get("industry", "")} industry
          - Their budget is {state.get("user_input", {}).get("budget", "")}
          - They mentioned these platforms: {analysis.get("mentioned_platforms", [])}
          - They have these marketing goals: {analysis.get("marketing_goals", [])}
          
          Make your question conversational, specific to their industry, and reference any relevant platforms or goals they mentioned.
          """

          try:
            response = llm.invoke(prompt)
            state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": response.content.strip()
            })
          except:
            # Fallback to a standard question
            state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": "Would you like to allocate a larger portion of your budget to Instagram ads?"
            })
        else:
          # For search ads or balanced approach, ask about campaign start date
          prompt = f"""
          Generate a personalized question asking when the user would like to start their marketing campaign.
          This is the final piece of information needed before I can generate the marketing plan.
          
          Consider these details:
          - Their business is in the {state.get("business_info", {}).get("industry", "")} industry
          - Their budget is {state.get("user_input", {}).get("budget", "")}
          - Their focus is on {user_preference}
          - They mentioned these platforms: {analysis.get("mentioned_platforms", [])}
          - They have these marketing goals: {analysis.get("marketing_goals", [])}
          
          Make your question conversational and specific to their industry, referencing any relevant platforms or goals they mentioned.
          Clearly indicate that providing the start date will allow you to proceed with generating the plan.
          For example: "Okay, we're almost ready! Just let me know when you'd like to start the campaign for your {state.get("business_info", {}).get("industry", "")} business, and I can draft your marketing plan."
          """

          try:
            response = llm.invoke(prompt)
            state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": response.content.strip()
            })
          except:
            # Fallback to a standard question
            state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": "When would you like to start your marketing campaign?"
            })
        state["current_stage"] = "refinement" # Move to refinement stage
        return state
      else:
        # If analysis is uncertain, ask for clarification
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = f"""
        Generate a friendly message asking the user to clarify their marketing focus preference.
        
        The message should:
        1. Acknowledge that you're not sure what they prefer
        2. Clearly present the three options: social media marketing, search ads, or a balanced approach
        3. Be conversational and helpful
        4. Be specific to their {state.get("business_info", {}).get("industry", "")} industry if possible
        
        Keep it brief but clear.
        """

        try:
          response = llm.invoke(prompt)
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": response.content.strip()
          })
        except:
          # Fallback to a standard clarification request
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "I'm not sure I understood your preference. Could you clarify if you'd like to focus on social media marketing, search ads, or would prefer a balanced approach with both?"
          })
        return state

    # STEP 5: Process Instagram budget allocation (if social media focus)
    # This is the 'refinement' stage
    if state["current_stage"] == "refinement" and \
       ("allocate a larger portion" in last_ai_content.lower() and "instagram ads" in last_ai_content.lower()) and \
       state.get("user_input", {}).get("focus") == "social media" and \
       not state.get("user_input", {}).get("start_date"):
      # Use intelligent analysis to understand Instagram allocation preference
      context_info = {
        "industry": state.get("business_info", {}).get("industry", ""),
        "budget": state.get("user_input", {}).get("budget", "")
      }
      analysis = analyze_user_response(message_content, context_info, "instagram_allocation")

      # Update budget allocation based on the analysis
      if analysis.get("increase_instagram", False):
        # User wants to increase Instagram allocation
        if not state.get("budget_allocation"):
          state["budget_allocation"] = {}

        # Use the specific percentage if provided, otherwise default to 50%
        instagram_percentage = analysis.get("specified_percentage", 50)
        state["budget_allocation"]["Instagram Ads"] = instagram_percentage
        logger.info(f"Setting Instagram budget allocation to {instagram_percentage}%")

      # If they suggested an alternative platform, note it
      if analysis.get("alternative_platform"):
        if not state.get("budget_allocation"):
          state["budget_allocation"] = {}
        # Assign a default percentage if not specified, or adjust existing
        state["budget_allocation"][analysis.get("alternative_platform")] = state["budget_allocation"].get(analysis.get("alternative_platform"), 40)
        logger.info(f"Noted alternative platform: {analysis.get('alternative_platform')}")

      # Generate a personalized question about campaign start date
      llm = ChatOpenAI(model="gpt-4o", temperature=0)
      prompt = f"""
      Generate a conversational question asking when the user would like to start their marketing campaign.
      This is the final piece of information needed before I can generate the marketing plan.
      
      Consider:
      - Their business is in the {state.get("business_info", {}).get("industry", "")} industry
      - They {'want' if analysis.get("increase_instagram", False) else "don't necessarily want"} to focus heavily on Instagram ads.
      - Their budget is {state.get("user_input", {}).get("budget", "")}
      
      If they mentioned concerns about Instagram ads or suggested alternative platforms, acknowledge that briefly.
      Make your question specific to their industry if possible, keeping it brief and friendly.
      Clearly indicate that providing the start date will allow you to proceed with generating the plan.
      For example: "Got it. One last thing: when would you like to start the campaign for your {state.get("business_info", {}).get("industry", "")} business? Once I have that, I can prepare your marketing plan."
      """

      try:
        response = llm.invoke(prompt)
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": response.content.strip()
        })
      except:
        # Fallback to a standard question
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "When would you like to start your marketing campaign?"
        })
      return state

    # STEP 6: Gather Campaign Start Date and Duration (Refinement Stage)
    if state["current_stage"] == "refinement":
      # Check if we are waiting for the start date
      if not state.get("user_input", {}).get("start_date"):
        logger.info("STEP 6: Waiting for campaign start date.")
        analysis_context = {"current_stage": "refinement"} 
        analysis = analyze_user_response(message_content, analysis_context, "campaign_start_date")
        
        is_affirmative_only = analysis.get("is_affirmative_only", False)
        provided_start_date = analysis.get("specific_date") or analysis.get("relative_timeframe") or analysis.get("seasonal_timing")
        provided_duration = analysis.get("campaign_duration")

        if is_affirmative_only and not provided_start_date and not provided_duration:
            # User just said 'yes' or similar without giving a date or duration.
            logger.info("STEP 6: User provided affirmative response but no specific start date. Re-prompting.")
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": "I understand you're ready to set a start date. Could you please provide a specific date or timeframe (e.g., 'next Monday', 'July 1st', 'in two weeks')?"
            })
        elif provided_start_date:
          state["user_input"]["start_date"] = provided_start_date
          logger.info(f"STEP 6: Captured campaign start date: {provided_start_date}")
          
          if provided_duration: # If duration was also captured along with start date
            state["user_input"]["campaign_duration"] = provided_duration
            logger.info(f"STEP 6: Captured campaign duration (along with start date): {provided_duration}")
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": f"Great! We'll set the campaign to start {state['user_input']['start_date']} and run for {state['user_input']['campaign_duration']}. Are you ready to generate the final marketing plan now?"
            })
          else: # Start date captured, now ask for duration
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": f"Okay, campaign start is set for {state['user_input']['start_date']}. How long should the campaign run (e.g., '3 months', '6 weeks')?"
            })
        elif provided_duration and not provided_start_date: # Duration given when start date was expected
            state["user_input"]["campaign_duration"] = provided_duration # Store it
            logger.info(f"STEP 6: Captured campaign duration: {provided_duration}, but start date is still missing. Re-prompting for start date.")
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": f"Got the campaign duration as {provided_duration}. When should the campaign start? Please provide a specific date or timeframe."
            })
        else: # No start date, not a simple affirmative (that case is handled above), so response was likely unclear or irrelevant for date.
          logger.info("STEP 6: No specific start date provided or response unclear. Re-prompting.")
          state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": "When would you like to start the campaign? Please provide a date or timeframe (e.g., 'next Monday', 'July 1st', 'in two weeks')."
          })
        return state

      # Check if we have start date but are waiting for duration
      elif state.get("user_input", {}).get("start_date") and not state.get("user_input", {}).get("campaign_duration"):
        logger.info("STEP 6: Have start date, waiting for campaign duration.")
        analysis_context = {"current_stage": "refinement", "start_date_already_set": True}
        analysis = analyze_user_response(message_content, analysis_context, "campaign_start_date") # Re-use, it extracts duration
        
        provided_duration = analysis.get("campaign_duration")
        is_affirmative_only = analysis.get("is_affirmative_only", False)
        # Check if they tried to change the start date again
        alternative_start_date = analysis.get("specific_date") or analysis.get("relative_timeframe") or analysis.get("seasonal_timing")

        if provided_duration:
          # Basic check for very generic affirmative as duration. More sophisticated checks could be added.
          # For example, if it's just "yes" and not like "yes, 2 weeks"
          if is_affirmative_only and provided_duration.lower() in ["yes", "ok", "okay", "sure", "fine", "good"]:
             is_meaningful_duration = False
             try:
                 # A quick check: does it contain numbers or typical duration words?
                 if any(char.isdigit() for char in provided_duration) or \
                    any(word in provided_duration.lower() for word in ["week", "month", "year", "day"]):
                     is_meaningful_duration = True
             except: pass # Ignore errors in this simple check
             
             if not is_meaningful_duration:
                logger.info(f"STEP 6: Affirmative response interpreted as duration ('{provided_duration}'), but it seems too vague. Re-prompting.")
                state["messages"].append({
                    "id": str(uuid.uuid4()),
                    "type": "ai",
                    "content": "I understand you're ready to set the duration. Could you please specify how long the campaign should run (e.g., '3 months', '6 weeks')?"
                })
                return state # Return early to prevent setting a vague duration

          state["user_input"]["campaign_duration"] = provided_duration
          logger.info(f"STEP 6: Captured campaign duration: {provided_duration}")
          state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": f"Great! We'll set the campaign to start {state['user_input']['start_date']} and run for {state['user_input']['campaign_duration']}. Are you ready to generate the final marketing plan now?"
          })
        elif alternative_start_date and alternative_start_date != state["user_input"]["start_date"]:
            logger.info(f"STEP 6: User provided a new start date '{alternative_start_date}' instead of duration.")
            state["user_input"]["start_date"] = alternative_start_date
            # Clear previously asked-for duration, as context changed.
            if "campaign_duration" in state["user_input"]:
                del state["user_input"]["campaign_duration"] 
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": f"Okay, updated campaign start to {state['user_input']['start_date']}. How long should the campaign run (e.g., '3 months', '6 weeks')?"
            })
        elif is_affirmative_only and not provided_duration and not alternative_start_date:
            # User just said 'yes' or similar to the question "How long should it run?"
            logger.info("STEP 6: User provided affirmative response but no specific duration. Re-prompting.")
            state["messages"].append({
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": "I understand you're ready to set the duration. Could you please specify how long the campaign should run (e.g., '3 months', '6 weeks')?"
            })
        else: # No duration provided, not an affirmative, not a new start date.
          logger.info("STEP 6: No specific campaign duration provided or response unclear. Re-prompting.")
          state["messages"].append({
              "id": str(uuid.uuid4()),
              "type": "ai",
              "content": "How long should the campaign run? For example, '3 months' or '6 weeks'."
          })
        return state

    # STEP 7: Generate final plan upon confirmation
    # This is also part of 'refinement' stage, leading to 'final'
    if state["current_stage"] == "refinement" and "generate" in last_ai_content and ("final" in last_ai_content or "plan" in last_ai_content):
      # Use intelligent analysis to understand if the user is confirming
      context_info = {
        "industry": state.get("business_info", {}).get("industry", ""),
        "focus": state.get("user_input", {}).get("focus", "")
      }
      analysis = analyze_user_response(message_content, context_info, "final_confirmation")

      if analysis.get("confirmed", False):
        state["current_stage"] = "final"
        # Generate the plan
        logger.info("User confirmed final plan generation")
        try:
          return generate_final_plan(state) # Call the graph node function
        except Exception as e:
          logger.error(f"Error generating final plan: {str(e)}")
          # Basic fallback
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "I encountered an error generating your plan. Please try again."
          })
          return state
      else:
        # Handle additional requests before generating the plan
        if analysis.get("requested_changes"):
          # User requested changes, acknowledge them
          changes = ", ".join(analysis.get("requested_changes"))
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": f"I understand you'd like to adjust {changes}. Let me know when you're ready for me to generate the final marketing plan."
          })
        elif analysis.get("needs_information"):
          # User asked for more information
          info_needed = ", ".join(analysis.get("needs_information"))
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": f"I'll be happy to provide more information about {info_needed}. After that, would you like me to generate the final marketing plan?"
          })
        elif analysis.get("hesitant"):
          # User seems hesitant
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "I understand you may have some hesitations. Is there anything specific you'd like to adjust before I generate the final marketing plan?"
          })
        else:
          # Generic confirmation request
          state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "Would you like me to generate the final marketing media plan now? Please confirm."
          })
        return state

    # STEP 8: Handle modifications after a plan has been generated
    if state["current_stage"] == "final":
      logger.info("STEP 8: Processing user input after plan generation (Stage: final)")
      context_info = {
          "budget_display": state.get("user_input", {}).get("budget", "unknown"),
          "start_date": state.get("user_input", {}).get("start_date", "unknown"),
          "campaign_duration": state.get("user_input", {}).get("campaign_duration", "unknown")
      }
      analysis = analyze_user_response(message_content, context_info, "plan_modification_request")

      if analysis.get("wants_budget_change") or analysis.get("wants_timeline_change"):
        logger.info("STEP 8: User expressed desire to modify the plan. Clearing old parameters and collecting new ones.")

        # Clear old modification-related parameters to ensure we collect fresh ones
        state["user_input"]["budget"] = None
        state["user_input"]["budget_value"] = None
        state["user_input"]["currency"] = None
        state["user_input"]["currency_symbol"] = None
        state["user_input"]["start_date"] = None
        state["user_input"]["campaign_duration"] = None
        
        current_inputs_log = []

        # Populate with any new values provided in THIS user message
        if analysis.get("new_budget_original_format"):
          state["user_input"]["budget"] = analysis["new_budget_original_format"]
          state["user_input"]["budget_value"] = analysis.get("new_budget_converted_standard_value") or analysis.get("new_budget_amount")
          state["user_input"]["currency"] = analysis.get("new_budget_currency", state.get("user_input",{}).get("currency_default", "USD")) # Keep a default or previous if any
          state["user_input"]["currency_symbol"] = analysis.get("new_budget_currency_symbol", state.get("user_input",{}).get("currency_symbol_default", "$"))
          logger.info(f"STEP 8: From user's current message, captured new budget: {state['user_input']['budget']}")
          current_inputs_log.append(f"new budget of {state['user_input']['budget']}")

        if analysis.get("new_start_date"):
          state["user_input"]["start_date"] = analysis["new_start_date"]
          logger.info(f"STEP 8: From user's current message, captured new start date: {state['user_input']['start_date']}")
          current_inputs_log.append(f"new start date of {state['user_input']['start_date']}")

        if analysis.get("new_campaign_duration"):
          state["user_input"]["campaign_duration"] = analysis["new_campaign_duration"]
          logger.info(f"STEP 8: From user's current message, captured new campaign duration: {state['user_input']['campaign_duration']}")
          current_inputs_log.append(f"new campaign duration of {state['user_input']['campaign_duration']}")

        # Now, determine what's STILL missing and ask for all of it
        missing_parts = []
        if not state["user_input"].get("budget"):
          missing_parts.append("the new budget")
        if not state["user_input"].get("start_date"):
          missing_parts.append("the new campaign start date")
        if not state["user_input"].get("campaign_duration"):
          missing_parts.append("the new campaign duration")

        ai_message_content = "Okay, you'd like to refine the plan. "
        if current_inputs_log:
            ai_message_content += f"I've noted your request for {', and '.join(current_inputs_log)}. "
        
        if missing_parts:
            ai_message_content += f"To regenerate the plan, please also provide {', and '.join(missing_parts)}."
            state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": ai_message_content})
            state["current_stage"] = "awaiting_plan_modification_details"
            logger.info(f"STEP 8: Moving to 'awaiting_plan_modification_details'. Asking for: {', '.join(missing_parts)}")
        else: # All three (budget, start_date, duration) were somehow provided in the single modification request
            logger.info("STEP 8: All modification details (budget, start date, duration) captured in one go.")
            state["marketing_channels"] = [] # Clear plan components for regeneration
            state["budget_allocation"] = {}
            state["ad_creatives"] = []
            confirmation_message = (
                f"Great! I'll regenerate the plan with the updated "
                f"budget: {state['user_input']['budget']}, "
                f"start date: {state['user_input']['start_date']}, "
                f"and campaign duration: {state['user_input']['campaign_duration']}. "
                f"Generating now..."
            )
            state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": confirmation_message})
            state["current_stage"] = "refinement"
            logger.info("STEP 8: All details provided. Set stage to 'refinement' for plan regeneration.")
            try:
                return generate_final_plan(state)
            except Exception as e:
                logger.error(f"Error regenerating final plan after modification in STEP 8: {str(e)}")
                state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": "I encountered an error while trying to regenerate your plan. Please try describing your changes again."})
                state["current_stage"] = "final" # Revert to final if regeneration fails
        return state

      # This condition is now stricter: user must be confirmed happy AND explicitly NOT want changes.
      elif (analysis.get("confirmed_happy_with_plan") and \
            not analysis.get("wants_budget_change") and \
            not analysis.get("wants_timeline_change")) or \
           analysis.get("requested_download_or_email"): # Requesting download/email is also a path to close this loop.
        logger.info("STEP 8: User is happy with the plan (and no pending changes identified) or requested download/email.")
        state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "Great! If you need anything else, just let me know. You can ask to download or email the plan again if you need to."
        })
        return state
      else:
        logger.info("STEP 8: User response after plan is unclear, asking for clarification on modification intent.")
        state["messages"].append({
            "id": str(uuid.uuid4()),
            "type": "ai",
            "content": "I'm not sure I understood. Are you happy with the current plan, or would you like to change the budget, campaign start date, or campaign duration to regenerate it? You can also ask to download or email the plan."
        })
        return state

    # Stage to handle collecting missing modification details over potentially multiple turns
    elif state["current_stage"] == "awaiting_plan_modification_details":
      logger.info("AWAITING_MOD_DETAILS: Processing user input for missing plan modification details.")
      # Use the same analysis type; it extracts any of budget, start_date, duration
      analysis = analyze_user_response(message_content, {}, "plan_modification_request")

      updated_in_this_turn_log = []
      # Populate with any new values IF THEY ARE CURRENTLY MISSING
      if not state["user_input"].get("budget") and analysis.get("new_budget_original_format"):
          state["user_input"]["budget"] = analysis["new_budget_original_format"]
          state["user_input"]["budget_value"] = analysis.get("new_budget_converted_standard_value") or analysis.get("new_budget_amount")
          state["user_input"]["currency"] = analysis.get("new_budget_currency", state.get("user_input",{}).get("currency_default", "USD"))
          state["user_input"]["currency_symbol"] = analysis.get("new_budget_currency_symbol", state.get("user_input",{}).get("currency_symbol_default", "$"))
          updated_in_this_turn_log.append(f"budget set to {state['user_input']['budget']}")
      
      if not state["user_input"].get("start_date") and analysis.get("new_start_date"):
          state["user_input"]["start_date"] = analysis["new_start_date"]
          updated_in_this_turn_log.append(f"start date set to {state['user_input']['start_date']}")

      if not state["user_input"].get("campaign_duration") and analysis.get("new_campaign_duration"):
          state["user_input"]["campaign_duration"] = analysis["new_campaign_duration"]
          updated_in_this_turn_log.append(f"campaign duration set to {state['user_input']['campaign_duration']}")
      
      if updated_in_this_turn_log:
          logger.info(f"AWAITING_MOD_DETAILS: From user's latest message, updated: {', '.join(updated_in_this_turn_log)}")

      # Check again what's still missing
      missing_parts = []
      if not state["user_input"].get("budget"):
        missing_parts.append("the new budget")
      if not state["user_input"].get("start_date"):
        missing_parts.append("the new campaign start date")
      if not state["user_input"].get("campaign_duration"):
        missing_parts.append("the new campaign duration")

      if missing_parts:
        ai_message_content = "Thanks for that information. "
        # Acknowledge what was just provided if anything, before asking for more.
        # This part could be made smarter, e.g. by looking at `updated_in_this_turn_log`
        # For now, a generic "Thanks!"
        ai_message_content += f"To regenerate the plan, I still need you to provide {', and '.join(missing_parts)}."
        state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": ai_message_content})
        logger.info(f"AWAITING_MOD_DETAILS: Still missing: {', '.join(missing_parts)}. Re-prompting.")
        return state
      else:
        # All three (budget, start date, duration) are now collected
        logger.info("AWAITING_MOD_DETAILS: All modification details (budget, start date, duration) now collected.")
        state["marketing_channels"] = [] # Clear plan components for regeneration
        state["budget_allocation"] = {}
        state["ad_creatives"] = []
        
        confirmation_message = (
            f"Excellent, all details received! I'll regenerate the plan with the "
            f"budget: {state['user_input']['budget']}, "
            f"start date: {state['user_input']['start_date']}, "
            f"and campaign duration: {state['user_input']['campaign_duration']}. "
            f"Generating now..."
        )
        state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": confirmation_message})
        state["current_stage"] = "refinement"
        logger.info("AWAITING_MOD_DETAILS: All details collected. Set stage to 'refinement' for plan regeneration.")
        try:
          return generate_final_plan(state)
        except Exception as e:
          logger.error(f"Error regenerating final plan after collecting all modification details: {str(e)}")
          state["messages"].append({"id": str(uuid.uuid4()), "type": "ai", "content": "I encountered an error while trying to regenerate your plan. Please try describing your changes again."})
          state["current_stage"] = "final" # Revert to final if regeneration fails
          return state

    # If execution reaches here, it means direct stage handling did not return.
    # This could be a new message after the plan is delivered, or an unhandled state.
    # Use the graph to process the current state.
    logger.info(f"Invoking LangGraph for stage: {state['current_stage']} based on current state logic.")
    result = marketing_agent.invoke(state)
    return result

  except Exception as e:
    logger.error(f"General error in on_message: {str(e)}", exc_info=True)
    # Add fallback response
    fallback_message = {
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": "I seem to be having trouble processing that. Could you please try again?"
    }
    # Ensure messages list exists before appending
    if "messages" not in state:
      state["messages"] = []
    state["messages"].append(fallback_message)
    logger.info(f"Adding fallback message due to general error: {fallback_message['id']}")

  return state

# Test function to directly run the LangGraph workflow
if __name__ == "__main__":
  import sys

  logger.info("=== Marketing Agent LangGraph Test ===")

  # Create initial state
  test_state: MarketingPlanState = {
    "messages": [],
    "business_info": {},
    "competitor_info": [],
    "marketing_channels": [],
    "budget_allocation": {},
    "ad_creatives": [],
    "user_input": {},
    "current_stage": "initial"
  }


  print("\n--- Initial State (Agent Asks for URL) ---")
  test_state = on_message(test_state, {"id": "init", "content": "hi"}) # Initial greeting from user
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.langchain.com" # Example URL
  print(f"\n--- User provides URL: {test_url} ---")
  test_state = on_message(test_state, {"id": "url-msg", "content": test_url})
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  print(f"\n--- User confirms industry (e.g., AI Software) ---")
  # Assuming the agent identified "AI Software" or similar and asked for confirmation.
  # We need to find the last AI message asking for confirmation.
  last_ai_q = [m['content'] for m in test_state['messages'] if m['type']=='ai'][-1]
  print(f"AI asked: {last_ai_q}")
  test_state = on_message(test_state, {"id": "industry-confirm", "content": "Yes, that's correct."})
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  print(f"\n--- User provides budget (e.g., $10000) ---")
  last_ai_q = [m['content'] for m in test_state['messages'] if m['type']=='ai'][-1]
  print(f"AI asked: {last_ai_q}")
  test_state = on_message(test_state, {"id": "budget-msg", "content": "$10000 per month"})
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  print(f"\n--- User provides marketing focus (e.g., balanced) ---")
  last_ai_q = [m['content'] for m in test_state['messages'] if m['type']=='ai'][-1]
  print(f"AI asked: {last_ai_q}")
  test_state = on_message(test_state, {"id": "focus-msg", "content": "A balanced approach sounds good."})
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  print(f"\n--- User provides campaign start date (e.g., next month) ---")
  last_ai_q = [m['content'] for m in test_state['messages'] if m['type']=='ai'][-1]
  print(f"AI asked: {last_ai_q}")
  test_state = on_message(test_state, {"id": "date-msg", "content": "Let's start next month."})
  for msg in test_state.get("messages", []): print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')}")

  print(f"\n--- User confirms final plan generation ---")
  last_ai_q = [m['content'] for m in test_state['messages'] if m['type']=='ai'][-1]
  print(f"AI asked: {last_ai_q}")
  test_state = on_message(test_state, {"id": "final-confirm-msg", "content": "Yes, generate the plan."})
  # The final plan might be long, so just print the last few messages
  print("Last few messages after final plan generation:")
  for msg in test_state.get("messages", [])[-3:]: print(f"{msg.get('type','unknown').upper()}: {msg.get('content','no_content')[:200]}...")


  logger.info(f"Final state stage after test: {test_state.get('current_stage', 'unknown')}")
  logger.info("=== Test Complete ===")

