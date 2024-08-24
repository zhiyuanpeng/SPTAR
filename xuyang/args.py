import argparse
from utils import reset_args

class PromptTuringArgs:
    def __init__(self) -> None:
        # default
        self.device = 'cuda:0'
        self.device_idx = '7'

        # llm model parameters
        self.llm_name = 'llama-v2-13b' # gpt2, llama-7b, vicuna-7b, llama-v2-13b
        if self.llm_name == 'llama-7b':
            self.model_name_or_path = "/ssd/public_datasets/llama/llama_to_hf/llama-7b"
            self.tokenizer_name_or_path = "/ssd/public_datasets/llama/llama_to_hf/llama-7b"
        elif self.llm_name == 'vicuna-7b':
            self.model_name_or_path = "/ssd/public_datasets/llama/vicuna-7b"
            self.tokenizer_name_or_path = "/ssd/public_datasets/llama/vicuna-7b"
        elif self.llm_name == 'llama-v2-13b':
            self.model_name_or_path = "/ssd/public_datasets/llama/llama_v2/Llama-2-13b-chat-hf"
            self.tokenizer_name_or_path = "/ssd/public_datasets/llama/llama_v2/Llama-2-13b-chat-hf"
        else:
            self.model_name_or_path = "gpt2"
            self.tokenizer_name_or_path = "gpt2"

        # peft_config
        self.peft_type = 'CAUSAL_LM'
        self.task_type = 'TEXT'
        self.num_virtual_tokens = 50
        self.prompt_tuning_init_text = "please generate query for this document"
        # self.prompt_tuning_init_text = "Does the passage answer the question? Please respond with a simple 'Yes' or 'No.'"
        self.peft_model_id = f"{self.llm_name}_{self.peft_type}_{self.task_type}"
        self.prompt_num = 2
        self.text_len = 1024

        # dataset parameters
        self.dataset_name = "ms_100"
        self.train_data = "/home/xwu/project/LLMsAgumentedIR/xuyang/data/ms_50/prompt_tuning_50_train_text.csv"
        self.eval_data = "/home/xwu/project/LLMsAgumentedIR/xuyang/data/ms_50/prompt_tuning_50_test_text.csv"
        self.test_data = "/home/xwu/project/LLMsAgumentedIR/xuyang/data/ms_50/prompt_tuning_50_test_text.csv"
        self.few_shot_num = 50
        self.fixed_prompt = True

        # prompt parameters
        self.max_length = 1024
        self.lr = 3e-2
        self.num_epochs = 100
        self.batch_size = 1
        self.eval_batch_size = 2

        # experiments parameters
        self.checkpoint_name = f"{self.dataset_name}_{self.model_name_or_path}_{self.peft_type}_{self.task_type}_v1.pt".replace("/", "_")
        
        # log file
        self.experiment_dir = '/home/xwu/project/SPTAR/xuyang/llm_models'
        self.experiment_description = 'test_pointwise_v1_{}_{}_{}_{}_{}'.format(self.dataset_name, self.peft_model_id, self.num_virtual_tokens, self.few_shot_num, self.prompt_num)


if __name__ == "__main__":
    base_args = PromptTuringArgs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_virtual_tokens", type=int, help="num virtual tokens for prompt")
    parser.add_argument("--llm_name", type=str, help="model name")
    parser.add_argument("--device_idx", type=str, help="device id")
    parser.add_argument("--prompt_num", type=int, help="prompt number")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--train_data", type=str, help="train data path")
    parser.add_argument("--eval_data", type=str, help="eval data path")
    parser.add_argument("--test_data", type=str, help="test data path")
    parser.add_argument("few_shot_num", type=int, help="few shot setting")
    args = parser.parse_args(namespace=base_args)
    args = reset_args(args)
    print(args)