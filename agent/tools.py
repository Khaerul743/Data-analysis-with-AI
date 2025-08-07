from langchain_core.tools import tool
from utils.data_format import beauty_output


class Tools:
    @staticmethod
    @tool
    def get_data():
        """Gunakan ini jika kamu ingin mengambil data dari user"""
        try:
            data = beauty_output("../data/pengeluaran.csv")
        except Exception as e:
            print(f"Terjadi kesalahan ketika agent memakai tool: {e}")
            return "Terjadi kesalahan saat mengambil data."
        return data


if __name__ == "__main__":
    tool = Tools()
    print(tool.get_data())
