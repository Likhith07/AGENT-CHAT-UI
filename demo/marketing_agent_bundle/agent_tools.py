from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
import json

# Custom tool for gathering website data using Tavily
class WebsiteAnalysisTool(BaseTool):
  name: str = "website_analysis"
  description: str = "Analyzes a business website to extract key information about the business."

  def _run(self, url: str) -> str:
    # Use Tavily to search for information about the business website
    search_tool = TavilySearchResults(k=5)

    # First, search for basic business info
    business_search = search_tool.invoke({
      "query": f"information about business at {url} including industry, products, services and target audience"
    })

    # Then search for competitor information
    competitor_search = search_tool.invoke({
      "query": f"major competitors of business at {url} and their marketing strategies"
    })

    # Combine the results
    result = {
      "business_info": business_search,
      "competitor_info": competitor_search
    }

    return json.dumps(result) 