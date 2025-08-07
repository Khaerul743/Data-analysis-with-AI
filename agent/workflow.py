from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from models import (
    AgentState,
    Columns,
    ColumnsStructuredOutput,
    MainAgentStructuredOutput,
)
from prompts import AgentPrompts
from tools import Tools

load_dotenv()

messages_result = {
    "messages": [
        HumanMessage(
            content="Tolong analisis data tersebut dengan output format rapi.",
            additional_kwargs={},
            response_metadata={},
            id="91dd09c8-7d56-4feb-8403-725ec842b297",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_RDUKaX2jtAUPhUPvjG629SGL",
                        "function": {"arguments": "{}", "name": "get_data"},
                        "type": "function",
                    }
                ],
                "refusal": None,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 10,
                    "prompt_tokens": 124,
                    "total_tokens": 134,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                },
                "model_name": "gpt-4o-2024-08-06",
                "system_fingerprint": "fp_a288987b44",
                "id": "chatcmpl-C1mV1x7OL3gf6Ui21VdtHWXlSpiht",
                "service_tier": "default",
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
            id="run--84452602-f5b4-4aac-81d1-514b01541d0f-0",
            tool_calls=[
                {
                    "name": "get_data",
                    "args": {},
                    "id": "call_RDUKaX2jtAUPhUPvjG629SGL",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 124,
                "output_tokens": 10,
                "total_tokens": 134,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0},
            },
        ),
        ToolMessage(
            content="       tanggal  jumlah           kategori\n0   2025-01-01   12000            lainnya\n1   2025-01-01   12000            makanan\n2   2025-01-02   60000            makanan\n3   2025-01-02   33000            makanan\n4   2025-01-02   50000            makanan\n5   2025-01-03   35000       transportasi\n6   2025-01-03   40000            makanan\n7   2025-01-04   37000            makanan\n8   2025-01-04   38000            makanan\n9    2025-01-5   39000            lainnya\n10   2025-01-5  100000  keperluan pribadi\n11   2025-01-5   41000  keperluan pribadi\n12   2025-01-6   42000            makanan\n13   2025-01-7   43000            makanan",
            name="get_data",
            id="183efeb1-2b08-4aaf-97a8-f4affde906b2",
            tool_call_id="call_RDUKaX2jtAUPhUPvjG629SGL",
        ),
    ],
    "user_query": "Tolong analisis data tersebut dengan output format rapi.",
    "data": "       tanggal  jumlah           kategori\n0   2025-01-01   12000            lainnya\n1   2025-01-01   12000            makanan\n2   2025-01-02   60000            makanan\n3   2025-01-02   33000            makanan\n4   2025-01-02   50000            makanan\n5   2025-01-03   35000       transportasi\n6   2025-01-03   40000            makanan\n7   2025-01-04   37000            makanan\n8   2025-01-04   38000            makanan\n9    2025-01-5   39000            lainnya\n10   2025-01-5  100000  keperluan pribadi\n11   2025-01-5   41000  keperluan pribadi\n12   2025-01-6   42000            makanan\n13   2025-01-7   43000            makanan",
    "data_description": "Data tersebut merupakan catatan transaksi keuangan yang mencakup informasi tentang tanggal transaksi, jumlah uang yang terlibat, dan kategori transaksi. Terdapat beberapa kategori transaksi seperti makanan, transportasi, lainnya, dan keperluan pribadi yang dicatat dalam data tersebut.",
    "column_description": [
        Columns(
            column_name="tanggal", column_description="Tanggal transaksi dilakukan"
        ),
        Columns(
            column_name="jumlah",
            column_description="Jumlah uang dalam transaksi tersebut",
        ),
        Columns(
            column_name="kategori",
            column_description="Kategori dari transaksi, seperti makanan, transportasi, lainnya, dan keperluan pribadi",
        ),
    ],
    "data_stats": None,
    "is_analyis": False,
    "insight": None,
    "can_answer": False,
}


class Agent:
    def __init__(self):
        self.llm_for_reasoning = ChatOpenAI(model="gpt-4o")
        self.llm_for_explanation = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompts = AgentPrompts()
        self.tools = Tools()
        self.build = self._build_workflow

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("main_agent", self._main_agent)
        graph.add_node("agent_analysis", self._agent_analysis_data)
        graph.add_node("get_data", ToolNode(tools=[self.tools.get_data]))
        graph.add_node("save_tool_message", self._save_tool_message)
        graph.add_node("agent_data_desc", self._agent_data_description)

        # graph.add_edge(START, "main_agent")
        # graph.add_conditional_edges(
        #     "main_agent", self._main_agent_router, {"yes": END, "no": "agent_analysis"}
        # )
        # graph.add_conditional_edges(
        #     "agent_analysis",
        #     self._should_continue,
        #     {"tool_call": "get_data", "end": END},
        # )
        # graph.add_edge("get_data", "save_tool_message")
        # graph.add_edge("save_tool_message", END)
        graph.add_edge(START, "agent_data_desc")
        graph.add_edge("agent_data_desc", END)
        return graph.compile()

    def _main_agent(self, state: AgentState) -> Dict[str, Any]:
        # agentPrompts = self.prompts.main_agent()
        # llm = self.llm_for_explanation.with_structured_output(MainAgentStructuredOutput)
        # human = HumanMessage(content=state.user_query)
        # response = llm.invoke(agentPrompts + state.messages + [human])
        # print(response)
        # return {
        #     "can_answer": response.can_answer,
        #     "the_answer": response.the_answer,
        # }
        print("==========MAIN AGENT==========")
        return {"can_answer": False}

    def _main_agent_router(self, state: AgentState) -> str:
        can_answer = state.can_answer
        if can_answer:
            return "yes"
        return "no"

    def _agent_analysis_data(self, state: AgentState) -> Dict[str, Any]:
        prompt = self.prompts.agent_analyst_data(state.is_analyis, None)
        llm = self.llm_for_reasoning.bind_tools([self.tools.get_data])
        human = HumanMessage(content=(state.user_query))
        response = llm.invoke(prompt + state.messages + [human])
        print(response)
        return {"messages": [human] + [response]}

    def _should_continue(self, state: AgentState):
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_call"
        return "end"

    def _save_tool_message(self, state: AgentState) -> Dict[str, Any]:
        last_message = state.messages[-1]
        if isinstance(last_message, ToolMessage):
            return {"data": last_message.content}
        return {"data": "Kosong"}

    def _agent_data_description(self, state: AgentState):
        prompt = self.prompts.agent_data_description(state.data)
        llm = self.llm_for_explanation.with_structured_output(ColumnsStructuredOutput)
        response = llm.invoke(prompt)
        print(response)
        return {
            "data_description": response.data_description,
            "column_description": response.columns,
        }

    def run(self, query: str):
        workflow = self.build()
        messages = {
            "messages": [],
            "user_query": query,
            "data": None,
            "columns": None,
            "column_descriptions": None,
            "data_stats": None,
            "insight": None,
            "is_analyis": False,
        }
        result = workflow.invoke(messages_result)
        return result


if __name__ == "__main__":
    wf = Agent()
    result = wf.run("Tolong analisis data tersebut dengan output format rapi.")
    print(result)
