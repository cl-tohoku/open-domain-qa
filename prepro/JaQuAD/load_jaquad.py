from datasets import load_dataset


class JaQuAD:
    def __init__(self):
        self.data = load_dataset("SkelterLabsInc/JaQuAD")

    def __doc__(self):
        return "https://huggingface.co/datasets/SkelterLabsInc/JaQuAD"

    def __arxiv__(self):
        return "https://arxiv.org/abs/2202.01764"



if __name__ == "__main__":
    dataset = JaQuAD()
