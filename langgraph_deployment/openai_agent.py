from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pprint import pprint
### Tavily Search Tool
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
import os
from dotenv import load_dotenv

load_dotenv()

# setting up environment variables
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

if openai_key is not None:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

if tavily_key is not None:
    os.environ["TAVILY_API_KEY"] = tavily_key
else:
    raise ValueError("TAVILY_API_KEY environment variable is not set.")

if langchain_key is not None:
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
else:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"

# state schema
class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# llm models
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)

def call_model(state: State):
    return {"messages":[llm.invoke(state["messages"])]}


def make_default_graph():
    graph_workflow = StateGraph(State)

    # nodes
    graph_workflow.add_node("agent", call_model)

    # edges
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent

# tools
# 1. arxiv tool
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500) # type: ignore
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# 2. wikipedia tool
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500) # type: ignore
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# 3. tavily search tool
tavily = TavilySearch()

# 4. Custom Functions for math calculations
# add function
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

# multiply function
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# divide function
@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [arxiv, wiki, tavily, add, multiply, divide]

llm_with_tools = llm.bind_tools(tools)

def call_model_with_tool(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: State):
    if state["messages"][-1].tool_calls: # type: ignore
        return "tools"
    else:
        return END


def make_graph_with_tools():
    
    graph_workflow = StateGraph(State)

    tool_node = ToolNode(tools)

    graph_workflow.add_node("agent", call_model_with_tool)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent



# agent = make_default_graph()
agent = make_graph_with_tools()
    


