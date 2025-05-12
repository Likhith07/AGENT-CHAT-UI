from langchain_openai import ChatOpenAI
import json
import logging

logger = logging.getLogger(__name__)

# Add this helper function to intelligently analyze user responses with LLM
def analyze_user_response(user_message, context_info, question_type):
  """
  Use LLM to intelligently analyze user responses based on context and question type.
  Returns structured information based on the type of question being answered.
  """
  logger.info(f"Analyzing user response for question type: {question_type}")

  llm = ChatOpenAI(model="gpt-4o", temperature=0)

  if question_type == "industry_confirmation":
    prompt = f"""
    Analyze this user response: "{user_message}"
    
    Context: The user was asked to confirm if their business is in the {context_info.get('industry', '')} industry.
    
    Determine if the user is:
    1. Confirming (agreeing that the industry is correct)
    2. Correcting (providing a different industry)
    3. Asking for clarification
    
    Return your analysis as a JSON object with these fields:
    - "confirmed": true/false - whether they confirmed the industry
    - "corrected_industry": null or string - the industry they provided if they corrected it
    - "needs_clarification": true/false - if they seem confused or asked for clarification
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "budget_extraction":
    prompt = f"""
    Analyze this user response about their marketing budget: "{user_message}"
    
    Consider the following:
    1. What currency is being used (USD, rupees, euros, etc.)
    2. Is there a specific amount mentioned (including Indian formats like lakhs, crores)
    3. Is the time period mentioned (monthly, yearly, quarterly, etc.)
    4. Any constraints or flexibility mentioned about the budget
    
    Pay special attention to Indian number formats:
    - 1 lakh = 100,000 (one hundred thousand)
    - 1 crore = 10,000,000 (ten million)
    - If user says "20 crores", that's 200,000,000 (200 million)
    - For Indian budgets, use ₹ as the currency symbol
    
    Return your analysis as a JSON object with these fields:
    - "amount": the numeric amount extracted from the message (as a number, without commas or currency symbols)
    - "currency": the currency mentioned or implied (USD, rupees, etc.)
    - "currency_symbol": the appropriate symbol (e.g., "$", "₹", "€")
    - "period": the time period mentioned (monthly, yearly, etc.)
    - "flexible": true/false - whether they indicated flexibility
    - "original_format": the original budget format as mentioned by user (e.g., "20 crores", "5 lakhs")
    - "converted_standard_value": the budget converted to standard notation (e.g., 200000000 for "20 crores")
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "marketing_focus":
    prompt = f"""
    Analyze this user message about marketing focus preferences: "{user_message}"
    
    Context: The user was asked if they prefer to focus on social media marketing, search ads, or a balanced approach.
    Their business is in the {context_info.get('industry', '')} industry with a budget of {context_info.get('budget', '')}.
    
    Deeply analyze their response to understand their true intent, considering:
    1. Do they explicitly mention social media platforms (Facebook, Instagram, TikTok, etc.)?
    2. Do they mention search platforms (Google, Bing, etc.) or SEO?
    3. Do they imply they want both or a balanced approach?
    4. Do they mention specific goals (brand awareness, conversions, etc.)?
    
    Return your analysis as a JSON object with these fields:
    - "primary_focus": "social_media", "search_ads", or "balanced"
    - "confidence": number between 0-1 indicating how confident you are
    - "mentioned_platforms": array of specific platforms mentioned
    - "marketing_goals": array of any marketing goals mentioned
    - "needs_clarification": true/false
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "instagram_allocation":
    prompt = f"""
    Analyze this user response about Instagram budget allocation: "{user_message}"
    
    Context: The user was asked if they'd like to allocate a larger portion of their budget to Instagram ads.
    Their business is in the {context_info.get('industry', '')} industry.
    
    Determine:
    1. Are they agreeing to increase Instagram budget?
    2. Are they providing a specific percentage or amount?
    3. Are they suggesting an alternative platform?
    4. Are they declining or expressing concern?
    
    Return your analysis as a JSON object with these fields:
    - "increase_instagram": true/false
    - "specified_percentage": null or number (if they mentioned a specific percentage)
    - "alternative_platform": null or string (if they suggested another platform)
    - "concerns": array of any concerns mentioned
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "campaign_start_date":
    prompt = f"""
    Analyze this user response: "{user_message}"
    The user was likely asked about when to start a marketing campaign or for campaign duration.
    
    Consider:
    1. Did they mention a specific date (e.g., "July 1st, 2024", "next Monday")?
    2. Did they mention a relative timeframe (e.g., "next week", "in a month", "ASAP", "now")?
    3. Did they mention a seasonal timing (e.g., "before holiday season", "summer")?
    4. Did they mention a campaign duration (e.g., "3 months", "for 6 weeks")?
    5. Are there any conditions they want met before starting?
    6. Is the response merely an affirmation (e.g., 'yes', 'okay', 'sounds good', 'let\'s do it') reacting to a question about timing, without providing any *new* specific date, timeframe, or duration information?
    
    Return your analysis as a JSON object with these fields:
    - "is_affirmative_only": true/false - true if the response is just an affirmation without new timing details.
    - "has_date": true/false - whether they provided any information that could be interpreted as a start date or timeframe. If "is_affirmative_only" is true and no actual timing like "now" is part of the affirmation, this should generally be false.
    - "specific_date": null or date string if a specific date was mentioned.
    - "relative_timeframe": null or string describing relative timing.
    - "seasonal_timing": null or string describing seasonal timing.
    - "campaign_duration": null or string describing the campaign length if mentioned.
    - "conditions": array of any conditions mentioned before starting.

    IMPORTANT: If "is_affirmative_only" is true because the user said something like 'yes' or 'okay' to a question like 'Shall we set the start date?', do NOT invent a date like 'now'. In such cases, `specific_date`, `relative_timeframe`, and `seasonal_timing` should be null, and `has_date` should be false.
    If the user says "yes, start now", then `is_affirmative_only` could be true (or false, as it contains "now"), `has_date` true, and `relative_timeframe` "now". Use your best judgment. The key is to avoid extracting a date from a simple 'yes'.
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "final_confirmation":
    prompt = f"""
    Analyze this user response about generating a final marketing plan: "{user_message}"
    
    Consider:
    1. Are they confirming they want the final plan?
    2. Are they requesting changes before generating the plan?
    3. Are they asking for more information?
    4. Are they expressing confusion or hesitation?
    
    Return your analysis as a JSON object with these fields:
    - "confirmed": true/false - whether they confirmed to generate the plan
    - "requested_changes": array of any changes they requested
    - "needs_information": array of any information they requested
    - "hesitant": true/false - whether they seem hesitant
    
    ONLY return the JSON object, nothing else.
    """

  elif question_type == "plan_modification_request":
    prompt = f"""
    Analyze this user response regarding changes to an existing marketing plan: "{user_message}"
    The user has already seen a marketing plan (and may have already refined it once or multiple times) and is now potentially asking for further modifications.
    
    Contextual Information (current plan details):
    - Current Budget: {context_info.get('budget_display', 'unknown')}
    - Current Timeline/Start Date: {context_info.get('start_date', 'unknown')}
    - Current Campaign Duration: {context_info.get('campaign_duration', 'unknown')}
    
    Determine if the user wants to:
    1. Change the marketing budget.
    2. Change the campaign timeline (start date or duration).
    3. Confirm they are happy with the plan (no changes).
    4. Ask for download/email (this should be handled by a different logic, but note if mentioned).
    
    If they want to change the budget, extract:
    - "new_budget_amount": numeric amount (e.g., 1000000 for 1 million)
    - "new_budget_currency": currency code (e.g., "USD", "INR")
    - "new_budget_currency_symbol": currency symbol (e.g., "$", "₹")
    - "new_budget_original_format": the budget as mentioned by user (e.g., "1 million dollars", "50 lakhs")
    - "new_budget_converted_standard_value": budget converted to standard notation for new_budget_amount.

    If they want to change the timeline, extract:
    - "new_start_date": string describing the new start date (e.g., "next month", "2025-12-01")
    - "new_campaign_duration": string describing the new campaign duration if mentioned (e.g., "2 months", "for 6 weeks")

    Return your analysis as a JSON object with these fields:
    - "wants_budget_change": true/false
    - "new_budget_amount": null or number
    - "new_budget_currency": null or string
    - "new_budget_currency_symbol": null or string
    - "new_budget_original_format": null or string
    - "new_budget_converted_standard_value": null or number 
    - "wants_timeline_change": true/false
    - "new_start_date": null or string
    - "new_campaign_duration": null or string
    - "confirmed_happy_with_plan": true/false 
    - "requested_download_or_email": true/false
    - "other_request": null or string (if they have a different request not covered above)
    
    CRITICALLY IMPORTANT INSTRUCTIONS:
    - Your PRIMARY GOAL is to detect if the user wants to change budget, start date, or campaign duration. 
    - If the user says phrases like "refine the plan", "modify it", "make changes", "I want to change something", or similar generic statements indicating a desire to alter the plan without specifying *what* to change yet: this signals an intent to modify core parameters. In this case, set `wants_budget_change` to `true` (as a general flag indicating a desire for modification), ensure `confirmed_happy_with_plan` is `false`, and leave specific fields like `new_budget_amount` as `null` if no details were given. The main agent will then ask for the specific details (budget, start date, duration).
    - ANY other indication of wanting to explore different values for budget, start date, or duration (e.g., "what if budget was X?", "how about Y for duration?", "let's try Z for start date") means you MUST set `wants_budget_change` or `wants_timeline_change` to true accordingly, and `confirmed_happy_with_plan` to `false`.
    - `confirmed_happy_with_plan` should ONLY be true if the user VERY EXPLICITLY states satisfaction AND makes NO mention of changing budget, start date, or duration (e.g., "The plan is perfect!", "I'm happy with this.", "No changes needed, it looks great."). If they say "Looks good, but change X", `confirmed_happy_with_plan` MUST be `false`.
    - If the user's response is ambiguous, does not clearly state satisfaction, and does not clearly ask for a budget/timeline change (and isn't a generic modification request as described above), then all of `wants_budget_change`, `wants_timeline_change`, and `confirmed_happy_with_plan` should typically be `false`. This will allow the main agent logic to ask for clarification.
    - Pay attention to Indian number formats for budget (lakhs, crores).
    
    For example, if user says "change budget to 1 million dollars and timeline to 2 months":
    {{ 
      "wants_budget_change": true, "new_budget_amount": 1000000, "new_budget_currency": "USD", "new_budget_currency_symbol": "$", "new_budget_original_format": "1 million dollars", "new_budget_converted_standard_value": 1000000,
      "wants_timeline_change": true, "new_start_date": null, "new_campaign_duration": "2 months",
      "confirmed_happy_with_plan": false, "requested_download_or_email": false, "other_request": null
    }}

    If user says "I want to modify the plan":
    {{ 
      "wants_budget_change": true, "new_budget_amount": null, "new_budget_currency": null, "new_budget_currency_symbol": null, "new_budget_original_format": null, "new_budget_converted_standard_value": null,
      "wants_timeline_change": false, "new_start_date": null, "new_campaign_duration": null,
      "confirmed_happy_with_plan": false, "requested_download_or_email": false, "other_request": null
    }}
    
    If user says "the plan looks good, email it to me":
    {{ 
      "wants_budget_change": false, "new_budget_amount": null, "new_budget_currency": null, "new_budget_currency_symbol": null, "new_budget_original_format": null, "new_budget_converted_standard_value": null,
      "wants_timeline_change": false, "new_start_date": null, "new_campaign_duration": null,
      "confirmed_happy_with_plan": true, "requested_download_or_email": true, "other_request": null
    }}

    If user says "What if we run it for 6 weeks instead?" after seeing a plan:
     {{ 
      "wants_budget_change": false, "new_budget_amount": null, "new_budget_currency": null, "new_budget_currency_symbol": null, "new_budget_original_format": null, "new_budget_converted_standard_value": null,
      "wants_timeline_change": true, "new_start_date": null, "new_campaign_duration": "6 weeks",
      "confirmed_happy_with_plan": false, "requested_download_or_email": false, "other_request": null
    }}

    If user says "Looks good, but let's change the start date to next Monday":
     {{ 
      "wants_budget_change": false, "new_budget_amount": null, "new_budget_currency": null, "new_budget_currency_symbol": null, "new_budget_original_format": null, "new_budget_converted_standard_value": null,
      "wants_timeline_change": true, "new_start_date": "next Monday", "new_campaign_duration": null,
      "confirmed_happy_with_plan": false, "requested_download_or_email": false, "other_request": null
    }}

    ONLY return the JSON object, nothing else.
    """
  else: # Should not happen with defined question_types
    logger.error(f"Unknown question type for analysis: {question_type}")
    return {}


  try:
    analysis_result = llm.invoke(prompt)
    response_text = analysis_result.content.strip()

    # Clean up any markdown or formatting
    if "```json" in response_text:
      response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
      response_text = response_text.split("```")[1].split("```")[0].strip()

    # Parse the response as JSON
    analysis = json.loads(response_text)
    logger.info(f"Analysis result for {question_type}: {analysis}")
    return analysis
  except Exception as e:
    logger.error(f"Error analyzing user response for {question_type}: {str(e)}")
    # Return a default object based on question type
    if question_type == "industry_confirmation":
      return {"confirmed": True, "corrected_industry": None, "needs_clarification": False}
    elif question_type == "budget_extraction":
      # Check for common Indian currency formats in the raw message
      if "crore" in user_message.lower() or "cr" in user_message.lower():
        # Attempt basic extraction of crore values
        import re
        match = re.search(r'(\d+)(?:\s*(?:crore|cr|crores))', user_message.lower())
        if match:
          crore_value = int(match.group(1))
          return {
            "amount": crore_value,
            "currency": "rupees",
            "currency_symbol": "₹",
            "period": "monthly",
            "flexible": False,
            "original_format": f"{crore_value} crores",
            "converted_standard_value": crore_value * 10000000 # 1 crore = 10 million
          }
      elif "lakh" in user_message.lower() or "lac" in user_message.lower():
        # Attempt basic extraction of lakh values
        import re
        match = re.search(r'(\d+)(?:\s*(?:lakh|lac|lakhs))', user_message.lower())
        if match:
          lakh_value = int(match.group(1))
          return {
            "amount": lakh_value,
            "currency": "rupees",
            "currency_symbol": "₹",
            "period": "monthly",
            "flexible": False,
            "original_format": f"{lakh_value} lakhs",
            "converted_standard_value": lakh_value * 100000 # 1 lakh = 100 thousand
          }

      return {"amount": None, "currency": "USD", "currency_symbol": "$", "period": "monthly", "flexible": False, "original_format": "", "converted_standard_value": None}
    elif question_type == "marketing_focus":
      return {"primary_focus": "balanced", "confidence": 0.5, "mentioned_platforms": [], "marketing_goals": [], "needs_clarification": True}
    elif question_type == "instagram_allocation":
      return {"increase_instagram": False, "specified_percentage": None, "alternative_platform": None, "concerns": []}
    elif question_type == "campaign_start_date":
      return {
          "is_affirmative_only": False, 
          "has_date": False, 
          "specific_date": None, 
          "relative_timeframe": None, 
          "seasonal_timing": None, 
          "campaign_duration": None,
          "conditions": []
      }
    elif question_type == "final_confirmation":
      return {"confirmed": False, "requested_changes": [], "needs_information": [], "hesitant": True}
    elif question_type == "plan_modification_request":
      return {
        "wants_budget_change": False, "new_budget_amount": None, "new_budget_currency": None, "new_budget_currency_symbol": None, "new_budget_original_format": None, "new_budget_converted_standard_value": None,
        "wants_timeline_change": False, "new_start_date": None, "new_campaign_duration": None,
        "confirmed_happy_with_plan": False, "requested_download_or_email": False, "other_request": None
      }
    return {} 