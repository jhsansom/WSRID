import llms
import torch

class WatermarkedLLM:

    def __init__(self, llm, seed):
        self.llm = llm
        self.alpha = 0.01 # Relative proportion of watermark in overall prob

        self.generate_watermark(seed)

    def generate_watermark(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        watermark = torch.rand((self.llm.vocab_size,), generator=g)
        self.watermark = torch.nn.functional.softmax(watermark)

    def apply_watermark(self, probs):
        return ((1 - self.alpha) * probs) + (self.alpha * self.watermark)

    def autoregress_ids(self, input_ids, gen_len=20):
        for i in range(gen_len):
            logits = self.llm.get_logits(input_ids)
            probs = self.llm.logits_to_probs(logits)
            probs = self.apply_watermark(probs)
            out_id = self.llm.decode(probs)

            token = self.llm.tokenizer.decode(out_id)
            print(f'Output at step {i} is {out_id}: {token}')

            input_ids = torch.cat((input_ids, out_id.unsqueeze(0)))

        return input_ids



if __name__ == '__main__':
    model_name = 'huggyllama/llama-7b'
    #model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
    llm = llms.LLM(model_name)

    SEED = 12345
    wllm = WatermarkedLLM(llm, SEED)

    test_str = 'Hello my name is '
    token_list = llm.str_to_tokenlist(test_str)
    id_list = llm.tokenlist_to_idlist(token_list)
    id_list = wllm.autoregress_ids(id_list)

    token_list = llm.idlist_to_tokenlist(id_list)
    out_str = llm.tokenlist_to_str(token_list)

    print(f'Final output: {out_str}')


