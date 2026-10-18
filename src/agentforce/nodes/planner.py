from pydantic import BaseModel, Field
from typing import List,Dict,Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command,interrupt
from ..state import AgentState

class RequirementAnalysis(BaseModel):

    goal: List[str] =Field(description="Core objectives identified from the request.")
    constraints: List[str] = Field(description="Technical or functional constraints.")
    is_clear: bool = Field(description="True if the request has enough detail to build, False if it needs clarification.")
    missing_questions: List[str] = Field(description="Questions to ask the user if is_clear is False.")

class FilePlanItem(BaseModel):

    file_path: str = Field(description="Relative file path, e.g., 'src/main.py'")
    responsibility: str = Field(description="What this file does.")

class ProjectPlan(BaseModel):

    thought_process: str = Field(description="Brief architectural design summary.")
    files: List[FilePlanItem] = Field(description="List of files to create.")

def get_planner_llm():
    return ChatOllama(model="gemma4:e2b", temperature=0.1)

def analyze_requirements_node(state: AgentState) -> dict:
    print("--- [Node: Analyze Requirements] ---")
    llm = get_planner_llm().with_structured_output(RequirementAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the user requirement. Identify goals, constraints, and check if critical details are missing."),
        ("human", "{user_request}{user_answer}")
    ])
    
    chain = prompt | llm
    user_answer=state["user_classification",""]
    result: RequirementAnalysis = chain.invoke({"user_request": state["user_request"],"user_answer":user_answer})
    
    return {
        "requirements": result.dict(),
        "missing_information": result.missing_questions if not result.is_clear else []
        
    }

def create_plan_node(state: AgentState) -> dict:
    print("--- [Node: Create Project Plan] ---")
    llm = get_planner_llm().with_structured_output(ProjectPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the analyzed requirements, design a clean modular file structure and plan."),
        ("human", "Request: {user_request}\nRequirements: {requirements}")
    ])
    
    chain = prompt | llm
    result: ProjectPlan = chain.invoke({
        "user_request": state["user_request"],
        "requirements": state.get("requirements", {})
    })
    
    return {
        "file_plan": [f.dict() for f in result.files]
    }

def ask_user_node(state:AgentState) -> dict :
    questions = state.get("missing_information", [])
    user_request=state.get('user_classification',[])
    if user_request:
        ans=[user_request]
    ans.append(interrupt({
        "question":questions
    }))
    return{
        "user_classification": ans
    }

