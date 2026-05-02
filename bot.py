import os
import time
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="Magicpin Vera Bot")
START = time.time()

# In-memory stores
contexts: Dict[tuple[str, str], Dict[str, Any]] = {}    # (scope, context_id) -> {version, payload}
conversations: Dict[str, List[Dict[str, Any]]] = {}     # conversation_id -> [turns]
last_trigger_times: Dict[tuple[str, str], float] = {}   # (merchant_id, trigger_id) -> last fired time

# Initialize OpenAI client for OpenRouter
# Requires OPENROUTER_API_KEY environment variable
api_key = os.environ.get("OPENROUTER_API_KEY", "dummy_key_for_testing")
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# --- Endpoints ---

@app.api_route("/v1/healthz", methods=["GET", "HEAD"])
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok", 
        "uptime_seconds": int(time.time() - START), 
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "arpit kumar", 
        "team_members": ["arpit kumar"], 
        "model": LLM_MODEL,
        "approach": "Structured OpenRouter LLM composer with explicit prompt engineering for 5-dimension scoring.", 
        "contact_email": "2005sehalarpit@gmail.com",
        "version": "1.0.0", 
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    # Ignore stale versions, but accept identical versions (idempotent)
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True, 
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    # Process up to a reasonable number of triggers to avoid timeouts (max 20)
    for trg_id in body.available_triggers[:10]:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: 
            continue
            
        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        
        category_slug = merchant.get("category_slug") if merchant else None
        category = contexts.get(("category", category_slug), {}).get("payload") if category_slug else None
        
        customer_id = trg.get("customer_id")
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None
        
        if not (merchant and category): 
            continue
            
        # Deduplication / frequency cap: don't fire same trigger for same merchant within short window
        last_fired = last_trigger_times.get((merchant_id, trg_id), 0)
        if time.time() - last_fired < 300: # 5 minutes cooldown
            continue
            
        try:
            # Generate the message using LLM
            message_data = generate_initial_message(category, merchant, trg, customer)
            
            # Record that we're firing it
            last_trigger_times[(merchant_id, trg_id)] = time.time()
            
            conversation_id = f"conv_{merchant_id}_{trg_id}_{int(time.time())}"
            actions.append({
                "conversation_id": conversation_id,
                "merchant_id": merchant_id, 
                "customer_id": customer_id,
                "send_as": message_data.get("send_as", "vera"), 
                "trigger_id": trg_id,
                "template_name": "vera_dynamic_v1",
                "template_params": [merchant['identity']['name']],
                "body": message_data.get("body", ""), 
                "cta": message_data.get("cta", "open_ended"),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": message_data.get("rationale", "Composed dynamically")
            })
            
            # Initialize conversation history
            conversations[conversation_id] = [
                {"from": "vera", "msg": message_data.get("body", "")}
            ]
            
        except Exception as e:
            print(f"Error generating message for {trg_id}: {e}")
            
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Initialize conversation history if it doesn't exist
    if body.conversation_id not in conversations:
        conversations[body.conversation_id] = []
        
    conv_history = conversations[body.conversation_id]
    
    # Simple Auto-reply detection heuristic:
    # If the exact same message was sent multiple times consecutively by the merchant
    merchant_messages = [msg["msg"] for msg in conv_history if msg["from"] == "merchant"]
    merchant_messages.append(body.message)
    if len(merchant_messages) >= 3 and len(set(merchant_messages[-3:])) == 1:
        # Detected auto-reply (same message 3 times)
        conversations[body.conversation_id].append({"from": body.from_role, "msg": body.message})
        return {
            "action": "end", 
            "rationale": "Detected repeated auto-reply pattern. Exiting gracefully."
        }

    # Add new message to history
    conversations[body.conversation_id].append({"from": body.from_role, "msg": body.message})
    
    # Retrieve merchant and category for context in reply
    merchant = None
    category = None
    if body.merchant_id:
        merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload")
        if merchant:
            category_slug = merchant.get("category_slug")
            category = contexts.get(("category", category_slug), {}).get("payload")
    
    try:
        reply_data = generate_reply(category, merchant, conv_history)
        action = reply_data.get("action", "send")
        
        if action == "send":
            conversations[body.conversation_id].append({"from": "vera", "msg": reply_data.get("body", "")})
            return {
                "action": "send", 
                "body": reply_data.get("body", ""), 
                "cta": reply_data.get("cta", "open_ended"),
                "rationale": reply_data.get("rationale", "Responding to user intent.")
            }
        elif action == "wait":
            return {
                "action": "wait",
                "wait_seconds": reply_data.get("wait_seconds", 1800),
                "rationale": reply_data.get("rationale", "Waiting per user request.")
            }
        else:
            return {
                "action": "end",
                "rationale": reply_data.get("rationale", "Ending conversation gracefully.")
            }
            
    except Exception as e:
        print(f"Error generating reply: {e}")
        return {
            "action": "end", 
            "rationale": f"Error during reply generation: {str(e)}. Failsafe end."
        }

# --- LLM Functions ---

def generate_initial_message(category, merchant, trigger, customer=None) -> Dict:
    is_customer_facing = bool(customer)
    
    sys_prompt = f"""You are the intelligent brain behind Vera, magicpin's merchant assistant.
Your task is to craft a WhatsApp message based on the provided data.
You MUST output ONLY a JSON object with the following keys:
- "body": The message text (string)
- "cta": The Call To Action type. Usually "open_ended", "YES/STOP", or "none" (string)
- "send_as": "vera" or "merchant_on_behalf" (string)
- "rationale": A short explanation of WHY you chose this message and how it hits the 5 metrics (string)

SCORE MAXIMIZATION RULES:
1. SPECIFICITY: Include exact numbers, dates, places, or citations from the data. Be hyper-specific. Never say "increase your sales", say "crossed 100 reviews" or "2,100-patient trial".
2. CATEGORY FIT: Emulate the professional tone of the category (e.g. peer clinical for dentists). Use category taboos as strict negative constraints.
3. MERCHANT FIT: Personalize! Mention their name, their performance data, their offers. Code-mix Hindi-English if their language preference implies it ("hi" or "hi-en mix").
4. TRIGGER RELEVANCE: Why now? You MUST mention the specific event/trigger in the opening.
5. ENGAGEMENT COMPULSION: Use curiosity, social proof, effort externalization, or loss aversion. Include a clear, simple CTA (e.g. Reply YES).

CONSTRAINTS:
- Keep it concise.
- DO NOT hallucinate facts, numbers, or sources.
- DO NOT put markdown formatting outside the JSON block. Return raw JSON.
"""

    user_prompt = f"""
=== CATEGORY CONTEXT ===
{json.dumps(category, indent=2)}

=== MERCHANT CONTEXT ===
{json.dumps(merchant, indent=2)}

=== TRIGGER CONTEXT ===
{json.dumps(trigger, indent=2)}

=== CUSTOMER CONTEXT ===
{json.dumps(customer, indent=2) if customer else "None (This is a Merchant-facing message)"}

Craft the message and return ONLY valid JSON.
"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Failed to parse JSON: {e}, Content: {content}")
        # Fallback
        return {
            "body": f"Hi {merchant['identity']['name']}, I noticed a new update. Would you like to know more? Reply YES.",
            "cta": "YES/STOP",
            "send_as": "merchant_on_behalf" if is_customer_facing else "vera",
            "rationale": "Fallback message due to parsing error."
        }


def generate_reply(category, merchant, history: List[Dict]) -> Dict:
    sys_prompt = f"""You are Vera, magicpin's merchant assistant, engaged in a conversation.
Based on the conversation history, decide the next action.
You MUST output ONLY a JSON object with the following keys:
- "action": "send", "wait", or "end" (string)
- "body": The message text if action is "send" (string)
- "cta": The Call To Action if "send" (string)
- "wait_seconds": Integer seconds if action is "wait" (integer, optional)
- "rationale": Short explanation of your decision (string)

BEHAVIOR RULES:
- INTENT TRANSITION: If the user says "yes/ok/let's do it", you MUST set "action" to "send", STOP qualifying, and START actioning. Provide a "body" like "Done! I've updated...", "Sending the draft...".
- HOSTILITY/UNSUBSCRIBE: If the user is hostile or asks to stop, apologize and use action "end".
- AUTO-REPLY/BOT: If the user sounds like an automated bot, gracefully exit and use action "end".
- OFF-TOPIC: Politely redirect to the core magicpin value.
- MATCH TONE: Match language (Hindi/English mix if appropriate).

CONSTRAINTS:
- Return ONLY valid JSON.
"""

    history_text = "\n".join([f"{msg['from'].upper()}: {msg['msg']}" for msg in history])
    
    user_prompt = f"""
=== MERCHANT CONTEXT ===
{json.dumps(merchant, indent=2) if merchant else "Unknown"}

=== CATEGORY CONTEXT ===
{json.dumps(category, indent=2) if category else "Unknown"}

=== CONVERSATION HISTORY ===
{history_text}

Decide the next step and return JSON.
"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Failed to parse JSON for reply: {e}, Content: {content}")
        return {
            "action": "end",
            "rationale": "Fallback end due to parsing error."
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=True)
