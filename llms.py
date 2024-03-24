from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HuggingFaceModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def str_to_tokenlist(self, input_str):
        return self.tokenizer.tokenize(input_str)

    def tokenlist_to_str(self, token_list):
        return self.tokenizer.convert_tokens_to_string(token_list)

    def tokenlist_to_idlist(self, token_list):
        return self.tokenizer.convert_tokens_to_ids(token_list)

    def idlist_to_tokenlist(self, id_list):
        return self.tokenizer.convert_ids_to_tokens(id_list)

    def get_logits(self, id_list):
        input_ids = torch.tensor([id_list])
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        return logits[0, -1, :]
