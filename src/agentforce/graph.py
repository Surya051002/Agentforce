from langgraph.graph import START,StateGraph,END
from .state import AgentGraph
from .nodes.planner import create_plan_node,analyze_requirements_node

builder=StateGraph(AgentGraph)

builder.add_node("analyze_requirements_node",analyze_requirements_node)
builder.add_node("create_plan_node",create_plan_node)
