from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from models import AgentState, MainAgentStructuredOutput
from prompts import AgentPrompts

load_dotenv()


class Agent:
    def __init__(self):
        self.llm_for_reasoning = ChatOpenAI(model="gpt-4o")
        self.llm_for_explanation = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompts = AgentPrompts()
        self.build = self._build_workflow

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("main_agent", self._main_agent)

        graph.add_edge(START, "main_agent")
        graph.add_edge("main_agent", END)

        return graph.compile()

    def _main_agent(self, state: AgentState) -> Dict[str, Any]:
        agentPrompts = self.prompts.main_agent()
        llm = self.llm_for_explanation.with_structured_output(MainAgentStructuredOutput)
        human = HumanMessage(content=state.user_query)
        response = llm.invoke(agentPrompts + state.messages + [human])
        print(response)
        return {"can_answer": response.can_answer, "the_answer": response.the_answer}

    def run(self, query: str):
        workflow = self.build()
        messages = {
            "messages": [],
            "user_query": query,
            "columns": None,
            "column_descriptions": None,
            "data_stats": None,
            "insight": None,
        }
        result = workflow.invoke(messages)
        return result


if __name__ == "__main__":
    wf = Agent()
    result = wf.run("Tolong analisis data tersebut dengan output format rapi.")
    print(result)
