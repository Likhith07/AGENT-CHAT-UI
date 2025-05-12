from typing import Dict, List, Any
import uuid
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from .agent_state import MarketingPlanState # Assuming agent_state.py is in the same directory

logger = logging.getLogger(__name__)

# Define the nodes for our graph
def initialize_state(state: MarketingPlanState) -> MarketingPlanState:
  """Initialize the state with default values."""
  logger.info("🔄 LangGraph Node: initialize_state - Starting")
  if "current_stage" not in state:
    state["current_stage"] = "initial"
    state["business_info"] = {}
    state["competitor_info"] = []
    state["marketing_channels"] = []
    state["budget_allocation"] = {}
    state["ad_creatives"] = []
    state["user_input"] = {}

    # Add a welcome message if this is a new state
    if "messages" not in state or not state["messages"]:
      # Use UUID format for message IDs for better compatibility with agent-chat-ui
      welcome_id = str(uuid.uuid4())
      state["messages"] = [{
        "id": welcome_id,
        "type": "ai",
        "content": "Welcome to the AI-Powered Marketing Media Plan Generator! Please provide your business website URL to start."
      }]
      logger.info(f"Generated welcome message with ID: {welcome_id}")

  logger.info(f"🔄 LangGraph Node: initialize_state - Completed. Current stage: {state['current_stage']}")
  return state

def extract_business_data(state: MarketingPlanState) -> MarketingPlanState:
  """Extract business data from the website URL."""
  logger.info("🔄 LangGraph Node: extract_business_data - Starting")
  # Get the last user message
  last_message = None
  for message in reversed(state["messages"]):
    if message.get("type") == "human":
      last_message = message
      break

  if not last_message:
    # Add a message asking for the website URL
    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": "Please provide the business website URL to analyze."
    })
    logger.info("🔄 LangGraph Node: extract_business_data - No user message found, requesting URL")
    return state

  # Check if the message contains a URL
  content = last_message.get("content", "")
  logger.info(f"🔄 LangGraph Node: extract_business_data - Processing message: {content[:30]}...")

  # Simple URL validation (could be more sophisticated)
  if "http" in content and "." in content:
    logger.info("🔄 LangGraph Node: extract_business_data - URL detected, analyzing website")
    try:
      logger.info("🔄 LangGraph Node: extract_business_data - Using direct analysis approach")

      # Use Tavily to get basic website information
      search_tool = TavilySearchResults(k=5)

      # Search for business information
      business_search = search_tool.invoke({
        "query": f"information about business at {content} including industry, products, services and target audience"
      })

      # Search for marketing strategies
      marketing_search = search_tool.invoke({
        "query": f"marketing strategies and social media presence of business at {content}"
      })

      # Use LLM to analyze the search results directly
      llm = ChatOpenAI(model="gpt-4o", temperature=0)

      analysis_prompt = f"""Analyze this business information for {content}:

      Business Information:
      {json.dumps(business_search, indent=2)}
      
      Marketing Information:
      {json.dumps(marketing_search, indent=2)}

      Create a comprehensive profile of the business with specific information about:
      1. Industry/Niche 
      2. Products/Services in detail
      3. Target Audience details (demographics, interests, etc.)
      4. Existing Marketing Strategies (social media presence, ads, etc.)

      Your response should be very structured and detailed.
      """

      logger.info("🔄 LangGraph Node: extract_business_data - Getting business analysis from LLM")
      analysis_result = llm.invoke(analysis_prompt)

      # Now extract structured data from the analysis
      struct_prompt = f"""Based on this business analysis, extract the following information as valid JSON:

      {analysis_result.content}

      Format your response EXACTLY like this with NO markdown formatting or additional text:
      {{
        "industry": "The industry name",
        "products": ["Product1", "Product2"],
        "target_audience": "Detailed description of target audience",
        "existing_marketing": "Description of marketing strategies"
      }}
      
      ONLY output the JSON object itself, no other text.
      """

      logger.info("🔄 LangGraph Node: extract_business_data - Extracting structured data from analysis")
      structured_result = llm.invoke(struct_prompt)

      # Clean and parse the JSON
      logger.info("🔄 LangGraph Node: extract_business_data - Parsing structured data")
      content_json = structured_result.content.strip()

      # Log the raw response to help with debugging
      logger.info(f"🔄 LangGraph Node: extract_business_data - Raw LLM JSON response: {content_json[:200]}...")

      # Try to clean up the response
      if content_json.startswith("```json"):
        content_json = content_json[7:]
      if content_json.startswith("```"):
        content_json = content_json[3:]
      if content_json.endswith("```"):
        content_json = content_json[:-3]
      content_json = content_json.strip()

      try:
        # Parse the JSON
        business_info = json.loads(content_json)
        state["business_info"] = business_info
        logger.info(f"🔄 LangGraph Node: extract_business_data - Business info extracted: {business_info.get('industry', 'unknown')} industry")

        # Add a message confirming the industry - exactly matching the example in the requirements
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": f"I found that your business is in the {business_info.get('industry', 'unknown')} industry. Is that correct?"
        })

        state["current_stage"] = "data_gathering"
        logger.info("🔄 LangGraph Node: extract_business_data - Moving to data_gathering stage")
      except json.JSONDecodeError:
        logger.error("JSON parsing error, using generic approach")
        # Use a generic approach that works for any industry
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "I've analyzed your website. Could you confirm what industry your business is in?"
        })
        state["current_stage"] = "data_gathering"

    except Exception as e:
      logger.error(f"Error in business data extraction: {str(e)}")
      # Generic fallback that works for any website/industry
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": "I had trouble analyzing your website. Could you tell me more about your business, including your industry and target audience?"
      })
      logger.info("🔄 LangGraph Node: extract_business_data - Using fallback approach due to error")
  else:
    # If no URL is found, ask for it
    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": "I need a valid business website URL to analyze. Please provide a URL starting with http:// or https://."
    })
    logger.info("🔄 LangGraph Node: extract_business_data - No valid URL found, requesting URL")

  return state

def gather_competitor_data(state: MarketingPlanState) -> MarketingPlanState:
  """Gather competitor data using Tavily API."""
  logger.info("🔄 LangGraph Node: gather_competitor_data - Starting")

  # Skip if we don't have business info yet
  if not state.get("business_info"):
    logger.info("🔄 LangGraph Node: gather_competitor_data - No business info available, skipping")
    return state

  # Get the industry from the business info
  industry = state["business_info"].get("industry", "")
  logger.info(f"🔄 LangGraph Node: gather_competitor_data - Industry identified: {industry}")

  if not industry:
    # Skip if we don't have the industry
    logger.info("🔄 LangGraph Node: gather_competitor_data - No industry specified, skipping")
    return state

  try:
    # Use Tavily to search for competitor information
    logger.info(f"🔄 LangGraph Node: gather_competitor_data - Searching for competitors in {industry} industry")
    search_tool = TavilySearchResults(k=5)

    # Get competitors
    query = f"top competitors in {industry} industry and their marketing strategies"
    logger.info(f"🔄 LangGraph Node: gather_competitor_data - Search query: {query}")
    search_results = search_tool.invoke({"query": query})

    # Get industry trends
    trend_query = f"recent trends and keywords in {industry} marketing"
    logger.info(f"🔄 LangGraph Node: gather_competitor_data - Trend search query: {trend_query}")
    trend_results = search_tool.invoke({"query": trend_query})

    # Process the search results with LLM
    logger.info("🔄 LangGraph Node: gather_competitor_data - Processing competitor search results with LLM")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Update the prompt in gather_competitor_data for more reliable JSON responses
    prompt = f"""Based on the following search results, identify the top competitors in the {industry} industry and their marketing strategies. 

    Competitor Search Results: 
    {json.dumps(search_results, indent=2)}

    Industry Trends:
    {json.dumps(trend_results, indent=2)}

    Format the response as a JSON array with the following structure:
    [
      {{
        "competitor_name": "Competitor Name",
        "ad_platforms": ["Platform1", "Platform2"],
        "audience": "Detailed description of their target audience",
        "budget_estimate": "Estimated marketing budget if available"
      }}
    ]
    
    Include at least 2-3 competitors with detailed information about their marketing approaches.
    
    IMPORTANT: Your entire response MUST be ONLY valid JSON with NO explanation text, markdown formatting, or additional commentary. Just output the raw JSON array itself, no other text.
    """

    result = llm.invoke(prompt)
    logger.info("🔄 LangGraph Node: gather_competitor_data - LLM response received")

    try:
      # Clean up and parse JSON response
      content = result.content.strip()
      logger.info(f"🔄 LangGraph Node: gather_competitor_data - Raw competitor data: {content[:100]}...")

      # Use robust JSON parsing approach
      try:
        # First attempt - direct parse
        competitor_info = json.loads(content)
      except json.JSONDecodeError:
        # If direct parse fails, try to extract JSON from markdown or text
        logger.info("🔄 LangGraph Node: gather_competitor_data - Direct JSON parse failed, trying to extract")

        # Remove markdown code block markers
        if "```json" in content:
          content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
          content = content.split("```")[1].split("```")[0].strip()

        # Try again with cleaned content
        try:
          competitor_info = json.loads(content)
        except json.JSONDecodeError:
          # If still failing, use a more aggressive approach to find JSON
          logger.info("🔄 LangGraph Node: gather_competitor_data - Extraction failed, requesting specific JSON from LLM")

          # Ask LLM to fix the JSON
          fix_prompt = f"""
          The following text should be valid JSON but has formatting issues:

          {content}

          Please reformat this as valid JSON without explanation. ONLY output the valid JSON array.
          """

          fix_result = llm.invoke(fix_prompt)
          fixed_content = fix_result.content.strip()

          # Try again with fixed content
          if "```json" in fixed_content:
            fixed_content = fixed_content.split("```json")[1].split("```")[0].strip()
          elif "```" in fixed_content:
            fixed_content = fixed_content.split("```")[1].split("```")[0].strip()

          competitor_info = json.loads(fixed_content)

      state["competitor_info"] = competitor_info
      logger.info(f"🔄 LangGraph Node: gather_competitor_data - Found {len(competitor_info)} competitors")

      # Add a message asking for the budget, exactly matching the example in requirements
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": "What is your monthly budget for marketing?"
      })

      state["current_stage"] = "analysis"
      logger.info("🔄 LangGraph Node: gather_competitor_data - Moving to analysis stage")
    except json.JSONDecodeError as e:
      logger.error(f"Error parsing competitor info: {str(e)}")
      # If parsing fails, add a generic message
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": "I've researched your industry competitors. What is your monthly budget for marketing?"
      })
      state["current_stage"] = "analysis"
      logger.info("🔄 LangGraph Node: gather_competitor_data - JSON parsing failed, moving to analysis stage")
  except Exception as e:
    logger.error(f"Error gathering competitor data: {str(e)}")
    # Add a message asking for the budget even if we failed to get competitor data
    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": "Let's focus on your marketing plan. What is your monthly budget for marketing?"
    })
    state["current_stage"] = "analysis"
    logger.info("🔄 LangGraph Node: gather_competitor_data - Error in competitor research, moving to analysis stage")

  logger.info("🔄 LangGraph Node: gather_competitor_data - Completed")
  return state

def analyze_marketing_channels(state: MarketingPlanState) -> MarketingPlanState:
  """Analyze and recommend marketing channels."""
  logger.info("🔄 LangGraph Node: analyze_marketing_channels - Starting")

  # Check for budget in user_input
  if state.get("user_input", {}).get("budget"):
    logger.info(f"🔄 LangGraph Node: analyze_marketing_channels - Using existing budget: {state['user_input']['budget']}")
    budget = state["user_input"]["budget"]
    currency = state.get("user_input", {}).get("currency", "dollars")

    # Now that we have the budget, analyze the marketing channels
    logger.info(f"🔄 LangGraph Node: analyze_marketing_channels - Analyzing marketing channels for budget: {budget} {currency}")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Get the industry for context-aware recommendations
    industry = state.get("business_info", {}).get("industry", "").lower()
    target_audience = state.get("business_info", {}).get("target_audience", "")

    analysis_prompt = f"""As a marketing expert, analyze this business and recommend the best marketing strategy:

    Business Information:
    {json.dumps(state["business_info"], indent=2)}

    Competitors:
    {json.dumps(state["competitor_info"], indent=2)}

    Monthly Budget: {budget} {currency}

    1. First, analyze this specific business and its industry context to understand what marketing approaches would work best.

    2. Based on your analysis, provide marketing channel recommendations in this JSON format:
    {{
      "recommended_channels": ["Channel1", "Channel2", "Channel3"],
      "industry_specific_strategy": "A paragraph explaining why these channels are particularly effective for this specific business type and industry",
      "budget_allocation": {{
        "Channel1": Percentage,
        "Channel2": Percentage,
        "Channel3": Percentage
      }},
      "ad_creatives": [
        {{
          "platform": "Channel1",
          "ad_type": "Type of ad",
          "creative": "Detailed and specific creative suggestion tailored to this business"
        }}
      ]
    }}

    The percentages in budget_allocation should add up to 100. Don't use generic channel names - provide specific platform recommendations (e.g., "LinkedIn Ads" instead of just "Social Media").

    Include at least 4-5 channels that are most appropriate for this specific business's industry and target audience.
    For the ad_creatives, provide detailed, industry-specific creative recommendations that would resonate with their particular audience.

    IMPORTANT: Your entire response MUST be ONLY valid JSON with NO explanation text, markdown formatting, or additional commentary. Just output the raw JSON object.
    """

    try:
      logger.info("🔄 LangGraph Node: analyze_marketing_channels - Requesting AI-powered recommendations")
      result = llm.invoke(analysis_prompt)

      # Clean up and parse JSON response
      content = result.content.strip()
      logger.info(f"🔄 LangGraph Node: analyze_marketing_channels - Raw recommendations: {content[:100]}...")

      # Handle potential JSON formatting issues with a more robust approach
      try:
        # First attempt - direct parse
        analysis = json.loads(content)
      except json.JSONDecodeError:
        # If direct parse fails, try to extract JSON from markdown or text
        logger.info("🔄 LangGraph Node: analyze_marketing_channels - Direct JSON parse failed, trying to extract")

        # Remove markdown code block markers
        if "```json" in content:
          content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
          content = content.split("```")[1].split("```")[0].strip()

        # Try again with cleaned content
        try:
          analysis = json.loads(content)
        except json.JSONDecodeError:
          # If still failing, use a more aggressive approach to find JSON
          logger.info("🔄 LangGraph Node: analyze_marketing_channels - Extraction failed, requesting specific JSON from LLM")

          # Ask LLM to fix the JSON
          fix_prompt = f"""
          The following text should be valid JSON but has formatting issues:

          {content}

          Please reformat this as valid JSON without explanation. ONLY output the valid JSON object.
          """

          fix_result = llm.invoke(fix_prompt)
          fixed_content = fix_result.content.strip()

          # Try again with fixed content
          if "```json" in fixed_content:
            fixed_content = fixed_content.split("```json")[1].split("```")[0].strip()
          elif "```" in fixed_content:
            fixed_content = fixed_content.split("```")[1].split("```")[0].strip()

          analysis = json.loads(fixed_content)

      state["marketing_channels"] = analysis.get("recommended_channels", [])
      state["budget_allocation"] = analysis.get("budget_allocation", {})
      state["ad_creatives"] = analysis.get("ad_creatives", [])
      state["industry_specific_strategy"] = analysis.get("industry_specific_strategy", "")

      logger.info(f"🔄 LangGraph Node: analyze_marketing_channels - AI recommended {len(state['marketing_channels'])} channels")

      # Use LLM to generate a natural question about focus preference
      next_question_prompt = f"""
      Based on these marketing recommendations for a business in the {industry} industry:
      {json.dumps(analysis, indent=2)}

      Create a natural conversational message that:
      1. Informs the user about the recommended channels
      2. Asks if they would prefer to focus more on social media marketing, search ads, or have a balanced approach with both

      Make the message conversational but concise.
      """

      question_result = llm.invoke(next_question_prompt)

      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": question_result.content.strip()
      })

      state["current_stage"] = "refinement"
      logger.info("🔄 LangGraph Node: analyze_marketing_channels - Moving to refinement stage")
      return state
    except Exception as e:
      logger.error(f"Error generating AI marketing recommendations: {str(e)}")

      # Use LLM to create fallback recommendations
      fallback_prompt = f"""
      Create generic marketing channel recommendations for a business in the {industry} industry.
      Format your response as valid JSON with these keys:
      1. recommended_channels (array of strings)
      2. budget_allocation (object with channel names as keys and percentage numbers as values)
      3. ad_creatives (array of objects with platform, ad_type, and creative keys)

      Keep your response concise and ensure it's valid JSON.
      """

      try:
        fallback_result = llm.invoke(fallback_prompt)
        fallback_content = fallback_result.content.strip()

        # Parse and clean the response
        if fallback_content.startswith("```json"):
          fallback_content = fallback_content[7:]
        if fallback_content.startswith("```"):
          fallback_content = fallback_content[3:]
        if fallback_content.endswith("```"):
          fallback_content = fallback_content[:-3]
        fallback_content = fallback_content.strip()

        fallback_analysis = json.loads(fallback_content)
        state["marketing_channels"] = fallback_analysis.get("recommended_channels", [])
        state["budget_allocation"] = fallback_analysis.get("budget_allocation", {})
        state["ad_creatives"] = fallback_analysis.get("ad_creatives", [])
      except:
        # If LLM fallback fails too, use minimal defaults
        state["marketing_channels"] = [
          "Social Media Marketing",
          "Search Engine Marketing",
          "Content Marketing",
          "Email Marketing"
        ]
        state["budget_allocation"] = {
          "Social Media Marketing": 30,
          "Search Engine Marketing": 30,
          "Content Marketing": 20,
          "Email Marketing": 20
        }

      # Add a message asking about focus preference
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": f"I've analyzed your business profile and budget ({budget} {currency}). Would you like to focus more on social media or search ads?"
      })

      state["current_stage"] = "refinement"
      logger.info("🔄 LangGraph Node: analyze_marketing_channels - Using AI-generated fallback recommendations")
      return state

  # If we don't have a budget in user_input, ask for it
  state["messages"].append({
    "id": str(uuid.uuid4()),
    "type": "ai",
    "content": "What is your monthly budget for marketing?"
  })

  logger.info("🔄 LangGraph Node: analyze_marketing_channels - Asking for budget")
  return state

def refine_marketing_plan(state: MarketingPlanState) -> MarketingPlanState:
  """Refine the marketing plan based on user feedback using AI."""
  logger.info("🔄 LangGraph Node: refine_marketing_plan - Starting")

  user_focus = state.get("user_input", {}).get("focus")
  user_start_date = state.get("user_input", {}).get("start_date")
  last_message_content = ""
  last_human_message = next((msg for msg in reversed(state.get("messages", [])) if msg.get("type") == "human"), None)

  if last_human_message:
    last_message_content = last_human_message.get("content", "").lower()

  # Use LLM to understand user input and decide next steps
  llm = ChatOpenAI(model="gpt-4o", temperature=0)

  # Create a context for the LLM with all relevant state information
  context = {
    "user_focus": user_focus,
    "user_start_date": user_start_date,
    "last_user_message": last_message_content,
    "business_info": state.get("business_info", {}),
    "marketing_channels": state.get("marketing_channels", []),
    "budget": state.get("user_input", {}).get("budget", ""),
    "currency": state.get("user_input", {}).get("currency", "dollars"),
    "current_stage": state.get("current_stage", "refinement")
  }

  # Get the last AI message to understand what was asked
  last_ai_messages = [msg for msg in state.get("messages", []) if msg.get("type") == "ai"]
  last_ai_message = last_ai_messages[-1] if last_ai_messages else None
  last_ai_content = last_ai_message.get("content", "") if last_ai_message else ""

  # --- Stage 1: Determine Marketing Focus if not set ---
  if not user_focus:
    analysis_prompt = f"""
    Based on this context:
    - Business: {context['business_info'].get('industry', 'unknown industry')}
    - Last AI message: "{last_ai_content}"
    - User's response: "{last_message_content}"
    
    Analyze the user's response and determine their marketing focus preference.
    Consider the full semantic meaning of their message, not just keywords.
    
    If they expressed interest primarily in social media marketing, respond with: "social media"
    If they expressed interest primarily in search advertising, respond with: "search ads"
    If they expressed interest in both approaches or a balanced strategy, respond with: "balanced"
    If their preference is unclear, respond with: "unclear"
    
    Return ONLY one of these four values: "social media", "search ads", "balanced", or "unclear"
    No explanation or additional text.
    """

    focus_result = llm.invoke(analysis_prompt)
    focus_preference = focus_result.content.strip().lower()

    if focus_preference in ["social media", "search ads", "balanced"]:
      logger.info(f"🔄 LangGraph Node: refine_marketing_plan - AI detected user preference: {focus_preference}")
      state["user_input"]["focus"] = focus_preference

      # Generate appropriate next question based on focus
      if focus_preference == "social media":
        next_prompt = f"""
        Create a natural follow-up question asking if the user would like to allocate a larger portion of their budget to Instagram ads.
        Make it conversational and specific to their {context['business_info'].get('industry', '')} industry.
        Keep it brief but friendly.
        """
      else:
        next_prompt = f"""
        Create a natural follow-up question asking when the user would like to start their marketing campaign.
        Make it conversational and specific to their {context['business_info'].get('industry', '')} industry.
        Keep it brief but friendly.
        """

      response = llm.invoke(next_prompt)

      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": response.content.strip()
      })
      return state
    else:
      # If unclear, ask for clarification
      clarification_prompt = f"""
      Create a message asking the user to clarify their marketing focus preference.
      Offer three clear options:
      1. Social media focus
      2. Search ads focus
      3. Balanced approach with both
      
      Make it conversational but keep it brief.
      """

      response = llm.invoke(clarification_prompt)

      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": response.content.strip()
      })
      return state

  # --- Stage 2: Handle Instagram Budget Allocation if focus is social media ---
  instagram_asked = any("instagram" in msg.get("content", "").lower() for msg in last_ai_messages)
  if user_focus == "social media" and instagram_asked and not user_start_date:
    # Use LLM to interpret the user's response about Instagram allocation
    instagram_prompt = f"""
    Based on the user's response: "{last_message_content}"
    
    Determine if they want to increase Instagram ad budget allocation.
    If they clearly said yes, return "yes"
    If they gave a specific percentage, return that percentage (just the number)
    If they said no or declined, return "no"
    If unclear, return "unclear"
    
    Return a single word or number.
    """

    instagram_result = llm.invoke(instagram_prompt)
    instagram_response = instagram_result.content.strip().lower()

    # Update budget allocation based on the response
    if instagram_response == "yes":
      # Default to 50% for Instagram
      if not state.get("budget_allocation"):
        state["budget_allocation"] = {}
      state["budget_allocation"]["Instagram Ads"] = 50
    elif instagram_response.isdigit() and 1 <= int(instagram_response) <= 100:
      # Use the specific percentage
      if not state.get("budget_allocation"):
        state["budget_allocation"] = {}
      state["budget_allocation"]["Instagram Ads"] = int(instagram_response)

    # Generate next question about campaign start date
    next_prompt = f"""
    Create a conversational question asking when the user would like to start their marketing campaign.
    Make it specific to their {context['business_info'].get('industry', '')} industry if possible.
    Keep it brief and friendly.
    """

    response = llm.invoke(next_prompt)

    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": response.content.strip()
    })
    return state

  # --- Stage 3: Determine Campaign Start Date ---
  if user_focus and not user_start_date:
    # Use LLM to analyze if the message contains a start date
    date_prompt = f"""
    Analyze this message: "{last_message_content}"
    
    Determine if it contains a campaign start date or timeframe.
    Return "yes" if it contains timing information like "next week", "January", a specific date, etc.
    Return "no" if it doesn't contain any date or time information.
    
    Return only "yes" or "no".
    """

    date_result = llm.invoke(date_prompt)
    has_date = date_result.content.strip().lower() == "yes"

    if has_date:
      # Store the original message as the start date
      state["user_input"]["start_date"] = last_message_content

      # Ask for final confirmation
      confirm_prompt = f"""
      Create a brief message asking if the user would like to generate the final marketing media plan now.
      Keep it conversational and brief.
      """

      response = llm.invoke(confirm_prompt)

      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": response.content.strip()
      })
    else:
      # Ask again about the start date
      campaign_questions = [msg for msg in state.get("messages", [])
                            if msg.get("type") == "ai" and "start" in msg.get("content", "").lower()
                            and "campaign" in msg.get("content", "").lower()]

      if len(campaign_questions) == 0:
        # First time asking
        date_question_prompt = f"""
        Create a friendly question asking when the user would like to start their marketing campaign.
        Suggest a few options like "next week", "next month", or a specific date.
        Make it conversational and brief.
        """

        response = llm.invoke(date_question_prompt)

        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": response.content.strip()
        })
      else:
        # Already asked before, default to next month
        state["user_input"]["start_date"] = "next month"

        # Ask for final confirmation
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "I'll set the campaign start date to next month. Would you like me to generate the final marketing media plan now?"
        })

      return state

  # --- Stage 4: Handle Final Confirmation ---
  if user_focus and user_start_date:
    # Check if the message is a confirmation
    confirm_prompt = f"""
    Based on the user's response: "{last_message_content}"
    
    Determine if they are confirming to generate the final marketing plan.
    Return "yes" if they're confirming (words like yes, sure, go ahead, generate, etc.)
    Return "no" if they're declining or asking for changes
    
    Return only "yes" or "no".
    """

    confirm_result = llm.invoke(confirm_prompt)
    is_confirmed = confirm_result.content.strip().lower() == "yes"

    if is_confirmed:
      state["current_stage"] = "final"
    else:
      # If not confirmed, ask again
      confirmation_questions = [msg for msg in state.get("messages", [])
                                if msg.get("type") == "ai" and "generate" in msg.get("content", "").lower()
                                and "final" in msg.get("content", "").lower()]

      if len(confirmation_questions) == 0:
        state["messages"].append({
          "id": str(uuid.uuid4()),
          "type": "ai",
          "content": "I have your preferences. Would you like me to generate the final marketing media plan now?"
        })
      else:
        # If already asked multiple times, assume yes
        state["current_stage"] = "final"

  return state

def generate_final_plan(state: MarketingPlanState) -> Dict[str, Any]:
  """Generate the final marketing media plan using AI intelligence."""
  logger.info("🔄 LangGraph Node: generate_final_plan - Starting")

  # Get the budget and currency
  budget = state.get("user_input", {}).get("budget", "")
  currency = state.get("user_input", {}).get("currency", "dollars")
  focus = state.get("user_input", {}).get("focus", "balanced")
  start_date = state.get("user_input", {}).get("start_date", "")
  campaign_duration = state.get("user_input", {}).get("campaign_duration", "not specified")

  # Create the structured media plan with available information
  media_plan = {
    "business_overview": {
      "industry": state["business_info"].get("industry", ""),
      "products": state["business_info"].get("products", []),
      "target_audience": state["business_info"].get("target_audience", ""),
      "existing_marketing": state["business_info"].get("existing_marketing", "")
    },
    "competitor_insights": state["competitor_info"],
    "recommended_channels": state["marketing_channels"],
    "budget_allocation": state["budget_allocation"],
    "suggested_ad_creatives": state["ad_creatives"],
    "user_input": state["user_input"]
  }

  # Using LLM to fill any gaps in the media plan based on industry knowledge
  llm = ChatOpenAI(model="gpt-4o", temperature=0)

  # If we're missing any key components, use AI to generate them
  if not state.get("marketing_channels") or not state.get("budget_allocation") or not state.get("ad_creatives"):
    complete_plan_prompt = f"""
    Create a comprehensive marketing plan for a business with these details:

    Industry: {state["business_info"].get("industry", "")}
    Target Audience: {state["business_info"].get("target_audience", "")}
    Products/Services: {state["business_info"].get("products", [])}
    Budget: {budget} {currency}
    Focus: {focus}
    Start Date: {start_date}
    Campaign Duration: {campaign_duration}

    Return your response as a JSON object with these keys:
    1. "recommended_channels" - array of string channel names appropriate for this specific industry
    2. "budget_allocation" - object with channel names as keys and percentage numbers as values (adding to 100%)
    3. "ad_creatives" - array of objects with "platform", "ad_type", and "creative" properties
    4. "industry_specific_strategy" - string containing specific strategic advice for this industry

    Use your knowledge of {state["business_info"].get("industry", "")} industry best practices to create highly specific, non-generic recommendations.

    IMPORTANT: Your entire response MUST be ONLY valid JSON with NO explanation text, markdown formatting, or additional commentary.
    """

    try:
      plan_result = llm.invoke(complete_plan_prompt)
      plan_content = plan_result.content.strip()

      # Use robust JSON parsing approach
      try:
        # First attempt - direct parse
        plan_data = json.loads(plan_content)
      except json.JSONDecodeError:
        # If direct parse fails, try to extract JSON from markdown or text
        logger.info("🔄 LangGraph Node: generate_final_plan - Direct JSON parse failed, trying to extract")

        # Remove markdown code block markers
        if "```json" in plan_content:
          plan_content = plan_content.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_content:
          plan_content = plan_content.split("```")[1].split("```")[0].strip()

        # Try again with cleaned content
        try:
          plan_data = json.loads(plan_content)
        except json.JSONDecodeError:
          # If still failing, use a more aggressive approach to find JSON
          logger.info("🔄 LangGraph Node: generate_final_plan - Extraction failed, requesting specific JSON from LLM")

          # Ask LLM to fix the JSON
          fix_prompt = f"""
          The following text should be valid JSON but has formatting issues:

          {plan_content}

          Please reformat this as valid JSON without explanation. ONLY output the valid JSON object.
          """

          fix_result = llm.invoke(fix_prompt)
          fixed_content = fix_result.content.strip()

          # Try again with fixed content
          if "```json" in fixed_content:
            fixed_content = fixed_content.split("```json")[1].split("```")[0].strip()
          elif "```" in fixed_content:
            fixed_content = fixed_content.split("```")[1].split("```")[0].strip()

          plan_data = json.loads(fixed_content)

      # Fill in any missing components
      if not state.get("marketing_channels"):
        media_plan["recommended_channels"] = plan_data.get("recommended_channels", [])

      if not state.get("budget_allocation"):
        media_plan["budget_allocation"] = plan_data.get("budget_allocation", {})

      if not state.get("ad_creatives") or len(state.get("ad_creatives", [])) == 0:
        media_plan["suggested_ad_creatives"] = plan_data.get("ad_creatives", [])

      # Add industry-specific strategy
      media_plan["industry_specific_strategy"] = plan_data.get("industry_specific_strategy", "")

    except Exception as e:
      logger.error(f"Error using AI to generate plan components: {str(e)}")

  # Generate the final marketing plan document using AI
  try:
    logger.info("🔄 LangGraph Node: generate_final_plan - Creating plan with AI")
    prompt = f"""
    You are a world-class marketing strategist with deep expertise in the {state["business_info"].get("industry", "")} industry.

    Create a professional, detailed marketing media plan based on this data:
    {json.dumps(media_plan, indent=2)}
    
    Format your response as a professional marketing document with these sections:

    1. Executive Summary - Brief overview of the plan
    2. Business Overview - Analysis of the business and its position
    3. Competitor Analysis - Insights about competitors and market position
    4. Marketing Strategy - Overall strategic approach specific to {state["business_info"].get("industry", "")} industry
    5. Channel Recommendations - Detailed breakdown of each marketing channel and why it's appropriate
    6. Budget Allocation - How the {budget} {currency} budget should be distributed
    7. Creative Direction - Specific ad creative recommendations for each platform
    8. Implementation Timeline - Starting {start_date} for a duration of {campaign_duration} with key milestones
    9. Performance Metrics - KPIs to track success

    Make this extremely specific to the {state["business_info"].get("industry", "")} industry with concrete, actionable recommendations.
    Include specific platforms, ad formats, and creative approaches that have proven effective in this industry.

    Format as markdown with proper headings and bullet points.
    """

    result = llm.invoke(prompt)

    # Add the structured JSON output
    state["messages"].append({
      "id": "final-plan-structured",
      "type": "ai",
      "content": json.dumps(media_plan, indent=2)
    })

    # Add the final plan to the messages
    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": result.content
    })

    # Generate a follow-up question about delivery using AI
    followup_prompt = f"""
    Create a friendly message asking if the user would like to download the marketing plan or have it emailed to them,
    or if they'd like to refine any part of the plan further.

    Keep it conversational and brief.
    """

    followup_result = llm.invoke(followup_prompt)

    state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": followup_result.content.strip()
    })

    logger.info("🔄 LangGraph Node: generate_final_plan - Plan generated successfully with AI")
  except Exception as e:
    logger.error(f"Error generating final plan with AI: {str(e)}")

    # Even the fallback uses AI to generate a response
    fallback_prompt = f"""
    Create a basic marketing plan for a {state["business_info"].get("industry", "")} business with a budget of {budget} {currency}.

    Format it as markdown with these sections:
    - Executive Summary
    - Marketing Channels
    - Budget Allocation
    - Implementation Timeline (starting {start_date} for {campaign_duration})
    - Success Metrics

    Keep it brief but professional and industry-specific.
    """

    try:
      fallback_result = llm.invoke(fallback_prompt)

      # Add the JSON structure
      state["messages"].append({
        "id": "final-plan-structured",
        "type": "ai",
        "content": json.dumps(media_plan, indent=2)
      })

      # Add the fallback plan
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": fallback_result.content
      })

      # Add final message
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": "Here's your marketing plan. Would you like to download it or have it emailed to you?"
      })
    except:
      # Ultimate fallback if everything else fails
      state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": f"I've created a basic marketing plan for your {state['business_info'].get('industry', '')} business with a budget of {budget} {currency}. Would you like me to refine any specific part of it?"
      })

  return state

# Add new function to handle plan delivery options
def handle_plan_delivery(state: MarketingPlanState) -> MarketingPlanState:
  """Handle the delivery of the final marketing plan."""
  logger.info("🔄 LangGraph Node: handle_plan_delivery - Starting")

  # Get the last message which should be the AI's generated plan
  final_plan_message = None
  if state.get("messages") and state["messages"][-1].get("type") == "ai":
    final_plan_message = state["messages"][-1].get("content", "Your marketing plan is ready.")
  else:
    # Fallback if the plan isn't the last message for some reason
    final_plan_message = "Your marketing plan has been generated."
    # Add the plan to messages if it wasn't there
    state["messages"].append({
        "id": str(uuid.uuid4()),
        "type": "ai",
        "content": final_plan_message # This might be just a placeholder if an error occurred
    })


  # Generate a follow-up message for plan delivery and next steps
  llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # Slightly more creative for a friendly closing
  
  prompt = f"""
  The marketing media plan has been generated. 
  Your current plan is:
  ---
  {final_plan_message[:1500]}... 
  ---
  
  Craft a friendly and helpful message to the user. This message should:
  1. Briefly acknowledge that the plan is ready.
  2. Offer options to download the plan or have it emailed.
  3. Crucially, ask if they are satisfied with the plan or if they would like to make any adjustments, specifically mentioning that they can change the **budget, campaign start date, or campaign duration**, which would regenerate the plan.
  4. Keep it concise and welcoming for further interaction.

  Example: 
  "Your marketing media plan is ready! You can download it now, or I can email it to you. 
  Are you happy with this plan, or would you like to make any adjustments? For example, we can change the **budget, campaign start date, or campaign duration**, and I'll regenerate the plan for you. Just let me know!"
  """

  try:
    response = llm.invoke(prompt)
    delivery_message_content = response.content.strip()
  except Exception as e:
    logger.error(f"Error generating plan delivery message: {str(e)}")
    delivery_message_content = "Your marketing plan is ready! Would you like to download it or have it emailed? If you'd like to make changes to the budget or timeline, let me know and I can regenerate it."

  # Add the delivery options message
  state["messages"].append({
      "id": str(uuid.uuid4()),
      "type": "ai",
      "content": delivery_message_content
  })

  logger.info("🔄 LangGraph Node: handle_plan_delivery - Plan delivery message added.")
  state["current_stage"] = "final" # Ensure stage is 'final' for next steps
  return state 