
from langgraph.graph import MessagesState
from typing import TypedDict,List,Annotated,Dict
from operator import add

class AgentState(TypedDict):

    Message: MessagesState
    user_request: List[str]
    user_classification: List[List[str]]
    requirements: Dict[str,any]
    file_plan: List[Dict]
    generated_files: List[str]
    test_results:bool
    retry_count: int
    max_retry: int

    