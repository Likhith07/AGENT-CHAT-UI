from typing import Dict, List, Any, TypedDict, Literal

# Define types
class MarketingPlanState(TypedDict):
  messages: List[Dict[str, Any]]
  business_info: Dict[str, Any]
  competitor_info: List[Dict[str, Any]]
  marketing_channels: List[str]
  budget_allocation: Dict[str, int]
  ad_creatives: List[Dict[str, Any]]
  user_input: Dict[str, Any]
  current_stage: Literal["initial", "data_gathering", "analysis", "refinement", "final"] 