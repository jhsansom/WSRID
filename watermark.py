import llms

class WatermarkedLLM:

    def __init__(self, llm):
        self.llm = llm

    def generate_watermark(self):
        pass

    

if __name__ == '__main__':
    llm = llms.HuggingFaceModel()