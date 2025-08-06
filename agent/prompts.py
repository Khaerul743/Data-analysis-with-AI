from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


class AgentPrompts:
    """prompts for the agent"""

    @staticmethod
    def main_agent(user_query: str) -> list[BaseMessage]:
        """Main agent prompt"""
        return [
            SystemMessage(
                content=(
                    """
            Kamu adalah asisten yang siap membantu pengguna.
            Tugas kamu adalah menjawab pertanyaan dari pengguna.
            Gunakan bahasa gaul yang santai dan mudah dimengerti.
"""
                )
            ),
            HumanMessage(content=user_query),
        ]

    @staticmethod
    def agent_analyst_data(
        user_query: str, is_analysis: bool, data_description
    ) -> list[BaseMessage]:
        data_description = f"""
        Berikut adalah deskripsi data yang sudah dianalisis:
        {data_description}
"""
        tools = "Jika user meminta analisis data, gunakan tools yang sesuai untuk melakukan analisis tersebut."
        return [
            SystemMessage(
                content=(
                    f"""
            Kamu adalah seorang data analyst profesional dengan pengalaman luas dalam analisis data.
            Tugas kamu adalah membantu pengguna dalam menganalisis data, menjawab pertanyaan terkait data yang disediakan.
            Berperilakulah seperti seorang data analyst profesional.
            {data_description if is_analysis else tools}
"""
                )
            ),
            HumanMessage(content=user_query),
        ]

    @staticmethod
    def agent_data_description(data) -> list[BaseMessage]:
        """Prompt for data description"""
        return [
            SystemMessage(
                content=(
                    """
            Kamu adalah seorang data analyst profesional yang akan memberikan deskripsi tentang data.
            Tugas kamu adalah menjelaskan data yang diberikan dengan jelas dan mudah dimengerti.
            Penjelasan berupa informasi tentang data tersebut.
"""
                )
            ),
            HumanMessage(
                content=f"""
            Berikut adalah data yang harus anda jelaskan:
            {data}
"""
            ),
        ]
