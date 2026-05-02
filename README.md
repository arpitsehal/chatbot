# Magicpin AI Challenge — Vera Bot

This repository contains the implementation of "Vera," the AI chatbot for the magicpin challenge. The bot engages and assists merchants on WhatsApp using the 4-context framework (Category, Merchant, Trigger, Customer).

## Architecture

- **Web Framework**: FastAPI (running on Uvicorn)
- **State Management**: In-memory dictionaries for context synchronization and conversation history.
- **LLM Engine**: OpenRouter API (`openai` python client)
- **LLM Model**: `openai/gpt-4o-mini` (configurable)

## Features

- Fully implements all 5 required endpoints (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`).
- Handles idempotency and versioning of context payloads.
- **Dynamic Prompt Engineering**: Leverages rigorous prompting techniques based directly on the judge rubric (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion).
- **Auto-reply Detection**: Automatically identifies repetitive automated messages and exits gracefully without burning tokens/turns.
- **Intent Handling**: Switches from "qualifying" mode to "action" mode when the merchant expresses agreement.

## Setup & Running Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenRouter API Key**:
   Set the `OPENROUTER_API_KEY` environment variable.
   
   **Windows (PowerShell)**:
   ```powershell
   $env:OPENROUTER_API_KEY="your-api-key-here"
   ```
   
   **macOS/Linux**:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

3. **Start the bot**:
   ```bash
   uvicorn bot:app --host 0.0.0.0 --port 8080
   ```

4. **Test the bot**:
   Open a new terminal window and run the judge simulator:
   ```bash
   python judge_simulator.py
   ```

## Design Decisions
- **JSON Object Responses**: We leverage OpenRouter's structured output mechanism (`response_format={ "type": "json_object" }`) to guarantee that the LLM returns the strictly required JSON shapes for messages and actions.
- **Fallback Mechanisms**: If the LLM call fails or times out, the bot safely reverts to sensible default values to avoid causing judge timeouts.
- **Deduplication**: Ensures the same trigger isn't fired repeatedly to the same merchant via a cooldown dictionary.
