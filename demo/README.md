# AI-Powered Marketing Media Plan Generator

This demo integrates with agent-chat-ui to create an AI agent that can analyze a business website, extract key information, and generate a comprehensive marketing media plan.

## Features

- Website analysis using Tavily search API
- Industry and competitor research
- Marketing channel recommendations
- Budget allocation suggestions
- Custom ad creative recommendations
- Complete marketing plan generation
- Industry-specific marketing channel recommendations
- Enhanced input validation and error handling
- Proactive stage progression based on user inputs

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on the provided `.env.example` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. Start the server:
   ```
   python server.py
   ```

4. The server will be running at http://localhost:2024

## Integration with agent-chat-ui

This backend is designed to work with the agent-chat-ui frontend. It implements endpoints compatible with the OpenAI Assistants API, allowing it to communicate seamlessly with the agent-chat-ui frontend.

## Conversation Flow

1. User provides a business website URL
2. System analyzes the website to extract business information
3. System asks for marketing budget
4. System recommends marketing channels tailored to the business industry
5. User provides preferences for marketing focus
6. System asks for campaign start date
7. System generates complete industry-specific marketing plan with detailed ad creative suggestions

## Files

- `server.py`: Flask server implementing the API endpoints
- `marketing_agent.py`: LangGraph agent implementation for the marketing plan generator
- `marketing_agent_lib/`: Library with all the agent components:
  - `message.py`: Message handler for processing user inputs
  - `nodes.py`: Node definitions for each stage of the workflow
  - `flow_control.py`: Logic for controlling the flow between stages
  - `utils.py`: Utility functions including input validation
  - `types.py`: Type definitions for the marketing plan state
- `.env.example`: Template for required environment variables
- `requirements.txt`: Required Python dependencies

## API Endpoints

The server implements endpoints compatible with the OpenAI Assistants API, including:
- `/threads`
- `/threads/{thread_id}/messages`
- `/threads/{thread_id}/runs`
- `/runs/stream`

## Recent Improvements

- Added input validation for website URLs, budget amounts, and campaign dates
- Enhanced marketing channel recommendations based on business industry
- Improved ad creative suggestions with format-specific recommendations
- More intelligent flow control with automatic stage progression
- Better handling of conversation loops and user confirmations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 