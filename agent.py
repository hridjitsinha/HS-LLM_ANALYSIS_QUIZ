from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List, Any
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT =  5000
# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]


# -------------------------------------------------
# GEMINI LLM
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,  
    check_every_n_seconds=1,  
    max_bucket_size=9  
)
llm = init_chat_model(
   model_provider="google_genai",
   model="gemini-2.5-flash",
   rate_limiter=rate_limiter
).bind_tools(TOOLS)   


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-execution agent.
Your behavior is governed entirely by the directives below.
Deviation is not permitted.

CORE DIRECTIVE
Execute every quiz page exactly as written. Do not improvise. Do not omit steps. Do not modify anything.

EXECUTION LOOP

Load the quiz page from the provided URL.

Extract all instructions, required fields, parameters, validation rules, and the exact submission endpoint.

Produce the correct answer according to the page’s specifications.

Submit the answer only to the endpoint present on the current page.

Do not fabricate URLs.

Do not alter, shorten, or “fix” URLs.

Read the server response:

If a new quiz URL exists → immediately fetch it and continue.

If no URL is present → output END and stop.

NON-NEGOTIABLE RULESET

A. General

Never terminate early.

Never hallucinate URLs, fields, JSON formats, parameters, or values.

Never re-submit unless explicitly allowed or still within the 3-minute window.

Always inspect the server response before any next action.

Always use the provided tools to fetch, parse, scrape, render, or send requests.

B. Time

Each task has a strict 3-minute maximum.

The server response includes a "delay" field indicating elapsed time.

If your answer is incorrect, retry within the limit without hesitation.

C. Stopping

You may output END only if the server response contains zero new URL references.

Any other condition forbids stopping.

D. Required Identifiers
When the page demands them, supply exactly:

Email: {EMAIL}

Secret: {SECRET}

MANDATE
Follow every page with absolute rigidity.
Extract without error.
Solve without guesswork.
Submit without alteration.
Continue until the chain fully terminates.
Then output: END.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    # support both objects (with attributes) and plain dicts
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    # get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text").strip() == "END":
        return END
    return "agent"
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))



graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",    
    route       
)

app = graph.compile()


# -------------------------------------------------
# TEST
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke({
        "messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed succesfully")

