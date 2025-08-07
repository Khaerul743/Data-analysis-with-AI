from utils.data_format import beauty_output


class Tools:
    def __init__(self):
        pass

    def get_data(self, path: str):
        """Gunakan ini jika kamu ingin mengambil data dari user"""
        try:
            data = beauty_output(path)
        except Exception as e:
            print(f"Terjadi kesalahan ketika agent memakai tool: {e}")
            return "Terjadi kesalahan saat mengambil data."
        return data


if __name__ == "__main__":
    tool = Tools()
    print(tool.get_data("../data/pengeluaran.csv"))
