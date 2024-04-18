from watermark import WatermarkedLLM
from datasegmenting import CreateDataset
from llms import LLM
import random

# Hyperparameters
NUM_TRIALS = 3
MODEL_NAME = 'openai-community/gpt2'
#MODEL_NAME = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
INIT_SENT = 'The following sentences are taken from the abstract of a scientific paper.'

human_evals = []
marked_evals = []
unmarked_evals = []

base_llm = LLM(MODEL_NAME)

for i in range(NUM_TRIALS):
    # Generate two LLMs, one watermarked and one not
    seed = random.randint(1, 9999)
    llm = WatermarkedLLM(base_llm, seed)

    # Loop over datapoints
    dataset = CreateDataset('sometext.txt', INIT_SENT)
    dataset_len = len(dataset)
    for j, datapoint in enumerate(dataset):
        # Extract datapoint
        prompt = datapoint['prompt']
        human_response = datapoint['rest']

        # Test whether our model classifies human response as human or AI-generated
        whole_str = prompt + ' ' + human_response
        id_list = llm.llm.str_to_idlist(whole_str)
        marked_logprob = llm.eval_log_prob(id_list)
        unmarked_logprob = llm.eval_log_prob(id_list, with_watermark=False)
        human_eval = (unmarked_logprob < marked_logprob) # Set to true if successful
        human_evals.append(human_eval)

        # Generate watermarked text
        id_list = llm.llm.str_to_idlist(prompt)
        marked_text = llm.autoregress_ids(id_list, gen_len=25, with_watermark=True)
        unmarked_text = llm.autoregress_ids(id_list, gen_len=25, with_watermark=False)

        # Test whether our model classifies watermarked text properly
        marked_text_produced = marked_text[len(id_list):]
        marked_logprob = llm.eval_log_prob(marked_text_produced)
        unmarked_logprob = llm.eval_log_prob(marked_text_produced, with_watermark=False)
        marked_eval = (unmarked_logprob > marked_logprob)
        marked_evals.append(marked_eval)

        # Test whether our model classifies non-watermarked text properly
        unmarked_text_produced = unmarked_text[len(id_list):]
        marked_logprob = llm.eval_log_prob(unmarked_text_produced)
        unmarked_logprob = llm.eval_log_prob(unmarked_text_produced, with_watermark=False)
        unmarked_eval = (unmarked_logprob < marked_logprob)
        unmarked_evals.append(unmarked_eval)

        # Print out results
        print(f'Iteration {j+1}/{dataset_len}: Human: {human_eval}, Marked: {marked_eval}, Unmarked: {unmarked_eval}', flush=True)

    human_correct = sum(human_evals)/len(human_evals)*100
    marked_correct = sum(marked_evals)/len(marked_evals)*100
    unmarked_correct = sum(unmarked_evals)/len(unmarked_evals)*100

    # Print everything
    print('='*50)
    print(f'Trial {i+1}/{NUM_TRIALS}')
    print(f'Human-generated text correctly classified: {human_correct:.2f}%')
    print(f'Watermarked text correctly classified: {marked_correct:.2f}%')
    print(f'Non-watermarked text correctly classified: {unmarked_correct:.2f}%')
    print('='*50)

