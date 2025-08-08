import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.llms import Replicate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from models import (
    AgentState,
    ColumnsStructuredOutput,
    MainAgentStructuredOutput,
)
from prompts import AgentPrompts
from tools import Tools
from utils.data_format import data_context_format

load_dotenv()


class Agent:
    def __init__(self):
        self.replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
        self.llm_for_reasoning = ChatOpenAI(model="gpt-4o")
        self.llm_for_explanation = ChatOpenAI(model="gpt-3.5-turbo")
        self.llm_for_classification = Replicate(
            model="ibm-granite/granite-3.2-8b-instruct",
            replicate_api_token=self.replicate_api_token,
        )
        self.memory = MemorySaver()
        self.prompts = AgentPrompts()
        self.tools = Tools()
        self.build = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("main_agent", self._main_agent)
        graph.add_node("agent_analysis", self._agent_analysis_data)
        graph.add_node("get_data", ToolNode(tools=[self.tools.get_data]))
        graph.add_node("save_tool_message", self._save_tool_message)
        graph.add_node("agent_data_desc", self._agent_data_description)
        graph.add_node("analysis_data", self._agent_analysis)
        graph.add_node("agent_insight", self._agent_data_insight)
        graph.add_node("agent_classification", self._agent_classification)

        graph.add_edge(START, "main_agent")
        graph.add_conditional_edges(
            "main_agent", self._main_agent_router, {"yes": END, "no": "agent_analysis"}
        )
        graph.add_conditional_edges(
            "agent_analysis",
            self._should_continue,
            {"tool_call": "get_data", "end": END},
        )
        graph.add_edge("get_data", "save_tool_message")
        graph.add_edge("save_tool_message", "agent_data_desc")
        graph.add_edge("agent_data_desc", "analysis_data")
        graph.add_edge("analysis_data", "agent_insight")
        graph.add_edge("agent_insight", "agent_classification")
        graph.add_edge("agent_classification", END)

        return graph.compile(checkpointer=self.memory)

    def _main_agent(self, state: AgentState) -> Dict[str, Any]:
        if state.is_analyis:
            return {"can_answer": False}
        agentPrompts = self.prompts.main_agent()
        llm = self.llm_for_explanation.with_structured_output(MainAgentStructuredOutput)
        human = HumanMessage(content=state.user_query)
        response = llm.invoke(agentPrompts + state.messages + [human])
        print(f"===============MAIN AGENT=============\n{response.the_answer}")
        return {
            "can_answer": response.can_answer,
            "the_answer": response.the_answer,
            "messages": state.messages
            + [human]
            + [AIMessage(content=response.the_answer)],
        }

    def _main_agent_router(self, state: AgentState) -> str:
        can_answer = state.can_answer
        if can_answer:
            return "yes"
        return "no"

    def _agent_analysis_data(self, state: AgentState) -> Dict[str, Any]:
        prompt = self.prompts.agent_analyst_data(state.is_analyis, state)
        llm = self.llm_for_reasoning.bind_tools([self.tools.get_data])
        human = HumanMessage(content=(state.user_query))
        response = llm.invoke(prompt + state.messages + [human])
        print(f"""
        ==============================PROMPT==========================
        {prompt}

        ==============================RESPONSE========================
        {response.content}
""")
        return {"messages": state.messages + [human] + [response]}

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
        print(f"===============AGENT DATA DESC=============\n{response}")
        return {
            "data_description": response.data_description,
            "column_description": response.columns,
        }

    def _agent_analysis(self, state: AgentState):
        analysisMean = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Kamu adalah seorang data analisis profesional yang memiliki pengalaman dalam menganalisis data.\nTugas kamu adalah membantu pengguna untuk menganalisis data yang telah mereka diberikan.\nUntuk output cukup berupa deskripsi dari data tersebut, kamu tidak perlu memberikan pertanyaan tambahan kepada pengguna.",
                ),
                (
                    "human",
                    "Tolong analisis data tersebut dengan benar. Berikut adalah detail dari data:\n- Data:\n{data}\n\n-Deskripsi dari data tersebut:\n{data_description}",
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        llm = self.llm_for_reasoning
        tools = [self.tools.analize_data]
        tool_calling = create_tool_calling_agent(
            llm=llm, tools=tools, prompt=analysisMean
        )
        executor_tool = AgentExecutor(agent=tool_calling, tools=tools, verbose=True)
        result = executor_tool.invoke(
            {"data": state.data, "data_description": state.data_description}
        )
        print(f"===============AGENT ANALYSIS=============\n{result}")
        return {"data_stats": result["output"], "is_analyis": True}

    def _agent_data_insight(self, state: AgentState):
        prompt = self.prompts.agent_insight_data(
            state.data, state.data_description, state.data_stats
        )
        llm = self.llm_for_explanation
        response = llm.invoke(prompt)
        print(f"===============INSIGHT=============\n{response.content}")
        return {"insight": response.content}

    def _agent_classification(self, state: AgentState):
        prompt = self.prompts.agent_classification(state.data)
        llm = self.llm_for_classification
        response = llm.invoke(prompt)
        print(f"===============AGENT CLASSIFICATION=============\n{response}")
        return {"data_classification": response}

    def run(self, state: AgentState, thread_id: str):
        return self.build.invoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )


if __name__ == "__main__":
    wf = Agent()
    thread = "thread_123"
    while True:
        user_input = input("Human: ")
        if user_input == "exit":
            break
        result1 = wf.run({"user_query": user_input}, thread)

    print(result1)

    # result2 = wf.run({"user_query": "Siapa nama saya?"}, thread)
    # print(result2)

    # result2 = wf.run

    # messages = {
    #     "messages": [],
    #     "user_query": query,
    #     "data": None,
    #     "columns": None,
    #     "column_descriptions": None,
    #     "data_stats": None,
    #     "insight": None,
    #     "is_analyis": False,
    # }

# messages_result = {
#     "messages": [
#         HumanMessage(
#             content="Tolong analisis data tersebut dengan output format rapi.",
#             additional_kwargs={},
#             response_metadata={},
#             id="91dd09c8-7d56-4feb-8403-725ec842b297",
#         ),
#         AIMessage(
#             content="",
#             additional_kwargs={
#                 "tool_calls": [
#                     {
#                         "id": "call_RDUKaX2jtAUPhUPvjG629SGL",
#                         "function": {"arguments": "{}", "name": "get_data"},
#                         "type": "function",
#                     }
#                 ],
#                 "refusal": None,
#             },
#             response_metadata={
#                 "token_usage": {
#                     "completion_tokens": 10,
#                     "prompt_tokens": 124,
#                     "total_tokens": 134,
#                     "completion_tokens_details": {
#                         "accepted_prediction_tokens": 0,
#                         "audio_tokens": 0,
#                         "reasoning_tokens": 0,
#                         "rejected_prediction_tokens": 0,
#                     },
#                     "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
#                 },
#                 "model_name": "gpt-4o-2024-08-06",
#                 "system_fingerprint": "fp_a288987b44",
#                 "id": "chatcmpl-C1mV1x7OL3gf6Ui21VdtHWXlSpiht",
#                 "service_tier": "default",
#                 "finish_reason": "tool_calls",
#                 "logprobs": None,
#             },
#             id="run--84452602-f5b4-4aac-81d1-514b01541d0f-0",
#             tool_calls=[
#                 {
#                     "name": "get_data",
#                     "args": {},
#                     "id": "call_RDUKaX2jtAUPhUPvjG629SGL",
#                     "type": "tool_call",
#                 }
#             ],
#             usage_metadata={
#                 "input_tokens": 124,
#                 "output_tokens": 10,
#                 "total_tokens": 134,
#                 "input_token_details": {"audio": 0, "cache_read": 0},
#                 "output_token_details": {"audio": 0, "reasoning": 0},
#             },
#         ),
#         ToolMessage(
#             content="       tanggal  jumlah           kategori\n0   2025-01-01   12000            lainnya\n1   2025-01-01   12000            makanan\n2   2025-01-02   60000            makanan\n3   2025-01-02   33000            makanan\n4   2025-01-02   50000            makanan\n5   2025-01-03   35000       transportasi\n6   2025-01-03   40000            makanan\n7   2025-01-04   37000            makanan\n8   2025-01-04   38000            makanan\n9    2025-01-5   39000            lainnya\n10   2025-01-5  100000  keperluan pribadi\n11   2025-01-5   41000  keperluan pribadi\n12   2025-01-6   42000            makanan\n13   2025-01-7   43000            makanan",
#             name="get_data",
#             id="183efeb1-2b08-4aaf-97a8-f4affde906b2",
#             tool_call_id="call_RDUKaX2jtAUPhUPvjG629SGL",
#         ),
#     ],
#     "user_query": "Tolong analisis data tersebut dengan output format rapi.",
#     "data": "       tanggal  jumlah           kategori\n0   2025-01-01   12000            lainnya\n1   2025-01-01   12000            makanan\n2   2025-01-02   60000            makanan\n3   2025-01-02   33000            makanan\n4   2025-01-02   50000            makanan\n5   2025-01-03   35000       transportasi\n6   2025-01-03   40000            makanan\n7   2025-01-04   37000            makanan\n8   2025-01-04   38000            makanan\n9    2025-01-5   39000            lainnya\n10   2025-01-5  100000  keperluan pribadi\n11   2025-01-5   41000  keperluan pribadi\n12   2025-01-6   42000            makanan\n13   2025-01-7   43000            makanan",
#     "data_description": "Data tersebut merupakan catatan transaksi keuangan yang mencakup informasi tentang tanggal transaksi, jumlah uang yang terlibat, dan kategori transaksi. Terdapat beberapa kategori transaksi seperti makanan, transportasi, lainnya, dan keperluan pribadi yang dicatat dalam data tersebut.",
#     "column_description": [
#         Columns(
#             column_name="tanggal", column_description="Tanggal transaksi dilakukan"
#         ),
#         Columns(
#             column_name="jumlah",
#             column_description="Jumlah uang dalam transaksi tersebut",
#         ),
#         Columns(
#             column_name="kategori",
#             column_description="Kategori dari transaksi, seperti makanan, transportasi, lainnya, dan keperluan pribadi",
#         ),
#     ],
#     "data_stats": "Berdasarkan analisis dari data transaksi tersebut, ditemukan bahwa:\n\n- Rata-rata transaksi (mean) adalah sekitar 41,571.43.\n- Nilai tengah (median) transaksi adalah 39,500.\n- Modus atau nilai yang paling sering muncul dalam data adalah 12,000.\n- Penyebaran data (standar deviasi) sebesar 20,240.39, menunjukkan variasi atau deviasi dari rata-rata.\n- Varian dari data adalah 409,673,469.39.\n- Nilai maksimum dari transaksi adalah 100,000, sedangkan nilai minimum adalah 12,000.\n- Jangkauan dari data transaksi ini adalah 88,000.\n- Distribusi data ini tidak terdistribusi secara normal, yang berarti data cenderung tidak simetris atau memiliki kurtosis yang jauh dari distribusi normal.\n\nData ini meliputi berbagai kategori seperti makanan, transportasi, dan keperluan pribadi yang memberikan gambaran umum tentang pengeluaran pada periode tersebut.",
#     "is_analyis": True,
#     "insight": "Dari data, deskripsi, dan hasil analisis statistik yang telah disediakan, terdapat beberapa insight penting yang dapat disimpulkan:\n\n1. **Pola Pengeluaran:**\n    - Mayoritas transaksi terjadi pada kategori makanan, diikuti oleh keperluan pribadi dan transportasi. Hal ini menunjukkan bahwa pengeluaran terbanyak pada periode tersebut adalah untuk makanan, yang kemungkinan merupakan kebutuhan sehari-hari.\n  \n2. **Rata-rata dan Distribusi Data:**\n    - Rata-rata transaksi sebesar 41,571.43 dengan nilai tengah 39,500. Data memiliki standar deviasi yang cukup tinggi sebesar 20,240.39, menunjukkan variasi yang signifikan dari rata-rata. Hal ini mengindikasikan adanya variasi pengeluaran yang cukup besar di dalam setiap kategori transaksi.\n\n3. **Modus dan Nilai Ekstrem:**\n    - Nilai yang paling sering muncul dalam data adalah 12,000, sedangkan nilai transaksi maksimum mencapai 100,000 dan minimum adalah 12,000. Terdapat perbedaan yang cukup jauh antara nilai transaksi maksimum dan minimum, menunjukkan variasi yang signifikan dari transaksi tertinggi hingga terendah.\n\n4. **Kesimpulan Distribusi Data:**\n    - Distribusi data tidak terdistribusi secara normal, menandakan bahwa data cenderung tidak simetris atau memiliki kurtosis yang jauh dari distribusi normal. Hal ini menunjukkan bahwa ada potensi adanya anomali atau pola pengeluaran yang tidak terduga dalam dataset tersebut.\n\nDengan demikian, insight-insight di atas dapat memberikan pemahaman yang lebih mendalam tentang pola pengeluaran, distribusi data, dan karakteristik transaksi keuangan pada periode yang diberikan. Ini dapat menjadi pijakan untuk analisis lebih lanjut atau pengambilan keputusan terkait pengeluaran di masa depan.",
#     "can_answer": False,
# }
