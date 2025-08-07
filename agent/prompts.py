from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


class AgentPrompts:
    """prompts for the agent"""

    @staticmethod
    def main_agent() -> list[BaseMessage]:
        """Main agent prompt"""
        return [
            SystemMessage(
                content=(
                    """
            Kamu adalah asisten yang siap membantu pengguna.
            Tugas kamu adalah menjawab pertanyaan simple dari pengguna.
            Gunakan bahasa gaul yang santai dan mudah dimengerti.
            
            Jika kamu disuruh untuk menganalisis sebuah data, response kamu tidak bisa jawab, dan serahkan tugas itu kepada agent analisis data.          
"""
                )
            )
        ]

    @staticmethod
    def agent_analyst_data(is_analysis: bool, data_description) -> list[BaseMessage]:
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
            )
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

    @staticmethod
    def agent_insight_data(data, data_description, data_stats) -> list[BaseMessage]:
        return [
            SystemMessage(
                content="""
                Kamu adalah seorang Data Scientist yang ahli dalam menemukan insight, pola, dan anomali dari data.
                Tugas kamu adalah membantu pengguna menyimpulkan insight dari data yang diberikan.
                Fokus pada interpretasi hasil statistik dan korelasikan dengan deskripsi data.
                Berikan insight yang mendalam dan jelas.
                """
            ),
            HumanMessage(
                content=f"""
                Saya telah menyediakan data, deskripsi, dan hasil analisis statistik.
                Tolong bantu saya menyimpulkan insight penting dari informasi ini:

                ### Data
                {data}

                ### Deskripsi Data
                {data_description}

                ### Hasil Analisis Statistik
                {data_stats}
                """
            ),
        ]


if __name__ == "__main__":
    prompt = AgentPrompts()
    print(prompt.agent_analyst_data(False, "woi"))
