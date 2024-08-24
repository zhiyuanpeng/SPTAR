from datasets import load_dataset
import torch
import random
from random import choice
from default_prompt import DefaultPrompt
random.seed(10)

class MSMARCODataset(object):
    def __init__(self, args, tokenizer) -> None:
        data_files = {"train": args.train_data, "test": args.eval_data}
        self.dataset = load_dataset("csv", data_files=data_files)
        self.test_dataset = load_dataset("csv", data_files={"test": args.test_data})
        self.args = args
        self.tokenizer = tokenizer
        self.text_column = "text_y"
        self.label_column = "text_x"
        self.max_length = args.max_length
        self.fixed_prompt = args.fixed_prompt
        self.prompt_from_train_only = False
        if self.fixed_prompt:
            if args.dataset_name == 'ms_50':
                self.fixed_one_shot_prompt = DefaultPrompt.ms_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
            if args.dataset_name == 'fiqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'hotpotqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'fever_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = []
        corpus_list = self.cut_text(examples[self.text_column], self.args.text_len)
        label_list =  self.cut_text(examples[self.label_column], self.args.text_len)
        if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
            full_prompt_idx = self.dataset['train'].shape[0]
            prompt_corpus_list = self.cut_text(self.dataset['train'][self.text_column], self.args.text_len)
            prompt_label_list = self.cut_text(self.dataset['train'][self.label_column], self.args.text_len)
        else:
            full_prompt_idx = batch_size
        if self.args.prompt_num == 2:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_one_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[j]))
        elif self.args.prompt_num == 3:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], corpus_list[j]))
        else:
            inputs = [f"Document: {x} \n Relevant Query: " for x in corpus_list]
        targets = [x for x in label_list]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        dynamic_input_length = 0
        dynamic_label_length = 0
        for i in model_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        for i in labels["input_ids"]:
            if len(i) > dynamic_label_length:
                dynamic_label_length = len(i)
        self.max_length = dynamic_input_length + dynamic_label_length + 5
        # if self.max_length >= 512:
        #     self.max_length = 512
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            if self.max_length >= len(sample_input_ids):
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            else:
                model_inputs["input_ids"][i] = sample_input_ids[-self.max_length:]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][-self.max_length:]
                labels["input_ids"][i] = label_input_ids[-self.max_length:]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def cut_text(self, examples, max_len=350):
        for i in range(len(examples)):
            if examples[i] is not None:
                max_length_text = len(examples[i].split(' '))
                if max_length_text > max_len:
                    examples[i] = ' '.join(examples[i].split(' ')[:max_len])
        return examples

    def test_preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"Document: {x} \n Relevant Query: " for x in examples[self.text_column]]
        model_inputs = self.tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
        return model_inputs
    
    def get_dataset(self):
        processed_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset['train'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        # eval_dataset = self.dataset["test"].map(
        #     self.test_preprocess_function,
        #     batched=True,
        #     num_proc=1,
        #     remove_columns=self.dataset["test"].column_names,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on dataset",
        # )
        test_dataset = self.test_dataset["test"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        return processed_datasets, test_dataset


class MSMARCODatasetWithNegCorpus(object):
    def __init__(self, args, tokenizer) -> None:
        self.train_dataset = load_dataset("csv", data_files={"train": args.train_data})
        self.eval_dataset = load_dataset("csv", data_files={"eval": args.eval_data})
        self.test_dataset = load_dataset("csv", data_files={"test": args.test_data})
        self.args = args
        self.tokenizer = tokenizer
        self.text_column = "pos_text_y"
        self.label_column = "text_x"
        self.pos_text_column = "pos_text_y"
        self.neg_text_column = "neg_text_y"
        self.max_length = args.max_length
        self.fixed_prompt = args.fixed_prompt
        self.prompt_from_train_only = False
        if self.fixed_prompt:
            if args.dataset_name == 'ms_50':
                self.fixed_one_shot_prompt = DefaultPrompt.ms_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
            if args.dataset_name == 'fiqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'hotpotqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'fever_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt

    def train_preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = []
        neg_inputs = []
        corpus_list = self.cut_text(examples[self.pos_text_column], self.args.text_len)
        neg_corpus_list = self.cut_text(examples[self.neg_text_column], self.args.text_len)
        label_list =  self.cut_text(examples[self.label_column], self.args.text_len)
        if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
            full_prompt_idx = self.dataset['train'].shape[0]
            prompt_corpus_list = self.cut_text(self.dataset['train'][self.text_column], self.args.text_len)
            prompt_label_list = self.cut_text(self.dataset['train'][self.label_column], self.args.text_len)
        else:
            full_prompt_idx = batch_size
        if self.args.prompt_num == 2:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_one_shot_prompt, corpus_list[j]))
                    neg_inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_one_shot_prompt, neg_corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], corpus_list[j]))
                        neg_inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], neg_corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[j]))
                        neg_inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], neg_corpus_list[j]))
        elif self.args.prompt_num == 3:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, corpus_list[j]))
                    neg_inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, neg_corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], corpus_list[j]))
                        neg_inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], neg_corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], corpus_list[j]))
                        neg_inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], neg_corpus_list[j]))
        else:
            inputs = [f"Document: {x} \n Relevant Query: " for x in corpus_list]
            neg_inputs = [f"Document: {x} \n Relevant Query: " for x in neg_corpus_list]
        targets = [x for x in label_list]
        model_inputs = self.tokenizer(inputs)
        model_neg_inputs = self.tokenizer(neg_inputs)
        labels = self.tokenizer(targets)
        neg_labels = self.tokenizer(targets)
        dynamic_input_length = 0
        dynamic_label_length = 0
        for i in model_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        for i in labels["input_ids"]:
            if len(i) > dynamic_label_length:
                dynamic_label_length = len(i)
        for i in model_neg_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        self.max_length = dynamic_input_length + dynamic_label_length + 5
        # if self.max_length >= 512:
        #     self.max_length = 512
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            neg_sample_input_ids = model_neg_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            neg_label_input_ids = neg_labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            model_neg_inputs["input_ids"][i] = neg_sample_input_ids + neg_label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            neg_labels["input_ids"][i] = [-100] * len(neg_sample_input_ids) + neg_label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            model_neg_inputs["attention_mask"][i] = [1] * len(model_neg_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            neg_sample_input_ids = model_neg_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            neg_label_input_ids = neg_labels["input_ids"][i]
            if self.max_length >= len(sample_input_ids):
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            else:
                model_inputs["input_ids"][i] = sample_input_ids[-self.max_length:]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][-self.max_length:]
                labels["input_ids"][i] = label_input_ids[-self.max_length:]
            if self.max_length >= len(neg_sample_input_ids):
                model_neg_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(neg_sample_input_ids)) + neg_sample_input_ids
                model_neg_inputs["attention_mask"][i] = [0] * (self.max_length - len(neg_sample_input_ids)) + model_neg_inputs["attention_mask"][i]
                neg_labels["input_ids"][i] = [-100] * (self.max_length - len(neg_sample_input_ids)) + neg_label_input_ids
            else:
                model_neg_inputs["input_ids"][i] = neg_sample_input_ids[-self.max_length:]
                model_neg_inputs["attention_mask"][i] = model_neg_inputs["attention_mask"][i][-self.max_length:]
                neg_labels["input_ids"][i] = neg_label_input_ids[-self.max_length:]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_neg_inputs["input_ids"][i] = torch.tensor(model_neg_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            model_neg_inputs["attention_mask"][i] = torch.tensor(model_neg_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
            neg_labels["input_ids"][i] = torch.tensor(neg_labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        model_neg_inputs["labels"] = neg_labels["input_ids"]
        model_inputs["neg_input_ids"] = model_neg_inputs['input_ids']
        model_inputs["neg_attention_mask"] = model_neg_inputs['attention_mask']
        model_inputs["neg_labels"] = model_neg_inputs['labels']
        return model_inputs

    def preprocess_function(self, examples):
        self.text_column = 'text_y'
        batch_size = len(examples[self.text_column])
        inputs = []
        corpus_list = self.cut_text(examples[self.text_column], self.args.text_len)
        label_list =  self.cut_text(examples[self.label_column], self.args.text_len)
        if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
            full_prompt_idx = self.dataset['train'].shape[0]
            prompt_corpus_list = self.cut_text(self.dataset['train'][self.text_column], self.args.text_len)
            prompt_label_list = self.cut_text(self.dataset['train'][self.label_column], self.args.text_len)
        else:
            full_prompt_idx = batch_size
        if self.args.prompt_num == 2:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_one_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[j]))
        elif self.args.prompt_num == 3:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], corpus_list[j]))
        else:
            inputs = [f"Document: {x} \n Relevant Query: " for x in corpus_list]
        targets = [x for x in label_list]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        dynamic_input_length = 0
        dynamic_label_length = 0
        for i in model_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        for i in labels["input_ids"]:
            if len(i) > dynamic_label_length:
                dynamic_label_length = len(i)
        self.max_length = dynamic_input_length + dynamic_label_length + 5
        # if self.max_length >= 512:
        #     self.max_length = 512
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            if self.max_length >= len(sample_input_ids):
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            else:
                model_inputs["input_ids"][i] = sample_input_ids[-self.max_length:]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][-self.max_length:]
                labels["input_ids"][i] = label_input_ids[-self.max_length:]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def cut_text(self, examples, max_len=512):
        for i in range(len(examples)):
            if examples[i] is not None:
                max_length_text = len(examples[i].split(' '))
                if max_length_text > max_len:
                    examples[i] = ' '.join(examples[i].split(' ')[:max_len])
        return examples

    def test_preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"Document: {x} \n Relevant Query: " for x in examples[self.text_column]]
        model_inputs = self.tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
        return model_inputs
    
    def get_dataset(self):
        train_dataset =  self.train_dataset["train"].map(
            self.train_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.train_dataset['train'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        eval_datasets = self.eval_dataset["eval"].map(
            self.train_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.eval_dataset['eval'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        test_dataset = self.test_dataset["test"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.test_dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        return train_dataset, eval_datasets, test_dataset

class MSMARCODatasetWithNegQuery(object):
    def __init__(self, args, tokenizer) -> None:
        self.train_dataset = load_dataset("csv", data_files={"train": args.train_data})
        self.eval_dataset = load_dataset("csv", data_files={"eval": args.eval_data})
        self.test_dataset = load_dataset("csv", data_files={"test": args.test_data})
        self.args = args
        self.tokenizer = tokenizer
        self.text_column = "text_y"
        self.label_column = "text_x"
        self.neg_label_column = "neg_text_x"
        self.max_length = args.max_length
        self.fixed_prompt = args.fixed_prompt
        self.prompt_from_train_only = False
        if self.fixed_prompt:
            if args.dataset_name == 'ms_50':
                self.fixed_one_shot_prompt = DefaultPrompt.ms_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
            if args.dataset_name == 'fiqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'hotpotqa_50':
                self.fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
            if args.dataset_name == 'fever_50':
                self.fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
                self.fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = []
        corpus_list = self.cut_text(examples[self.text_column], self.args.text_len)
        label_list =  self.cut_text(examples[self.label_column], self.args.text_len)
        neg_label_list = self.cut_text(examples[self.neg_label_column], self.args.text_len)
        if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
            full_prompt_idx = self.dataset['train'].shape[0]
            prompt_corpus_list = self.cut_text(self.dataset['train'][self.text_column], self.args.text_len)
            prompt_label_list = self.cut_text(self.dataset['train'][self.label_column], self.args.text_len)
        else:
            full_prompt_idx = batch_size
        if self.args.prompt_num == 2:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_one_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[j]))
        elif self.args.prompt_num == 3:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], corpus_list[j]))
        else:
            inputs = [f"Document: {x} \n Relevant Query: " for x in corpus_list]
        targets = [x for x in label_list]
        neg_targets = [x for x in neg_label_list]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        neg_labels = self.tokenizer(neg_targets)
        dynamic_input_length = 0
        dynamic_label_length = 0
        for i in model_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        for i in labels["input_ids"]:
            if len(i) > dynamic_label_length:
                dynamic_label_length = len(i)
        self.max_length = dynamic_input_length + dynamic_label_length + 5
        # if self.max_length >= 512:
        #     self.max_length = 512
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            if self.max_length >= len(sample_input_ids):
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            else:
                model_inputs["input_ids"][i] = sample_input_ids[-self.max_length:]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][-self.max_length:]
                labels["input_ids"][i] = label_input_ids[-self.max_length:]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
            neg_labels["input_ids"][i] = [-100] * (len(labels["input_ids"][i])-len(neg_labels["input_ids"][i])) + neg_labels["input_ids"][i]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["neg_labels"] = neg_labels["input_ids"]
        return model_inputs
    
    def cut_text(self, examples, max_len=350):
        for i in range(len(examples)):
            if examples[i] is not None:
                max_length_text = len(examples[i].split(' '))
                if max_length_text > max_len:
                    examples[i] = ' '.join(examples[i].split(' ')[:max_len])
        return examples

    def test_preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"Document: {x} \n Relevant Query: " for x in examples[self.text_column]]
        model_inputs = self.tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
        return model_inputs
    
    def get_dataset(self):
        train_dataset =  self.train_dataset["train"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.train_dataset['train'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        eval_datasets = self.eval_dataset["eval"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.eval_dataset['eval'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        test_dataset = self.test_dataset["test"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.test_dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        return train_dataset, eval_datasets, test_dataset
    
class MSMARCODatasetTestRandomFixInfer(MSMARCODataset):
    def __init__(self, args, tokenizer, prompt_list, index_id) -> None:
        super().__init__(args, tokenizer)
        if self.fixed_prompt:
            # if 'ms_' in args.dataset_name:
            if len(index_id) == 1:
                self.fixed_one_shot_prompt = "Document: {} \n Relevant Query: {} \n ".format(prompt_list[0]['corpus'], prompt_list[0]['query'])
            if len(index_id) == 2:
                self.fixed_two_shot_prompt = "Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n".format(prompt_list[0]['corpus'], prompt_list[0]['query'], prompt_list[1]['corpus'], prompt_list[1]['query'])
            # if args.dataset_name == 'fiqa_50':
            #     self.fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
            #     self.fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
            # if args.dataset_name == 'hotpotqa_50':
            #     self.fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
            #     self.fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
            # if args.dataset_name == 'fever_50':
            #     self.fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
            #     self.fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt

# Does the passage answer the query? Please respond with a simple "Yes" or "No."
class MSMARCOPointWiseDataset(object):
    def __init__(self, args, tokenizer) -> None:
        data_files = {"train": args.train_data, "test": args.eval_data}
        self.dataset = load_dataset("csv", data_files=data_files)
        self.test_dataset = load_dataset("csv", data_files={"test": args.test_data})
        self.args = args
        self.tokenizer = tokenizer
        self.text_column = "text_y"
        self.label_column = "text_x"
        self.score_column = "score"
        self.max_length = args.max_length
        self.fixed_prompt = args.fixed_prompt
        self.prompt_from_train_only = False
        if self.fixed_prompt:
            if args.dataset_name == 'ms_50':
                self.fixed_one_shot_prompt = """Passage: Long used as a folk medicine for prostate health, pumpkin seed oil reduces the size of an enlarged prostate, especially in the instance of benign prostatic hyperplasia (age-related prostate enlargement). (16, 17) Thatâs why I pumpkin seed oil use is one of three steps to improve prostate health!
                Query: what is pumpkin seed oil capsules good for
                Yes"""
                self.fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
        #     if args.dataset_name == 'fiqa_50':
        #         self.fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
        #         self.fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
        #     if args.dataset_name == 'hotpotqa_50':
        #         self.fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
        #         self.fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
        #     if args.dataset_name == 'fever_50':
        #         self.fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
        #         self.fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = []
        corpus_list = self.cut_text(examples[self.text_column], self.args.text_len)
        label_list =  self.cut_text(examples[self.label_column], self.args.text_len)
        if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
            full_prompt_idx = self.dataset['train'].shape[0]
            prompt_corpus_list = self.cut_text(self.dataset['train'][self.text_column], self.args.text_len)
            prompt_label_list = self.cut_text(self.dataset['train'][self.label_column], self.args.text_len)
        else:
            full_prompt_idx = batch_size
        if self.args.prompt_num == 2:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    # inputs.append("{} \n Passage: {} \n Question: {} \n".format(self.fixed_one_shot_prompt, corpus_list[j], label_list[j]))
                    inputs.append("Passage: {} \n Question: {} \n".format(corpus_list[j], label_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[j]))
        elif self.args.prompt_num == 3:
            for j in range(len(corpus_list)):
                if self.fixed_prompt:
                    inputs.append("{} \n Document: {} \n Relevant Query: ".format(self.fixed_two_shot_prompt, corpus_list[j]))
                else:
                    if (self.prompt_from_train_only and batch_size != self.dataset['train'].shape[0]):
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(prompt_corpus_list[prompt_idx_1], prompt_label_list[prompt_idx_1], prompt_corpus_list[prompt_idx_2], prompt_label_list[prompt_idx_2], corpus_list[j]))
                    else:
                        prompt_idx_1 = choice([m for m in range(0,full_prompt_idx) if m not in [j]])
                        prompt_idx_2 = choice([m for m in range(0,full_prompt_idx) if m not in [j, prompt_idx_1]])
                        inputs.append("Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".
                                    format(corpus_list[prompt_idx_1], label_list[prompt_idx_1], corpus_list[prompt_idx_2], label_list[prompt_idx_2], corpus_list[j]))
        else:
            inputs = [f"Document: {x} \n Relevant Query: " for x in corpus_list]
        targets = []
        for x in examples[self.score_column]:
            if x == 1:
                targets.append("Yes")
            else:
                targets.append("No")
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        dynamic_input_length = 0
        dynamic_label_length = 0
        for i in model_inputs["input_ids"]:
            if len(i) > dynamic_input_length:
                dynamic_input_length = len(i)
        for i in labels["input_ids"]:
            if len(i) > dynamic_label_length:
                dynamic_label_length = len(i)
        self.max_length = dynamic_input_length + dynamic_label_length + 5
        # if self.max_length >= 512:
        #     self.max_length = 512
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            if self.max_length >= len(sample_input_ids):
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            else:
                model_inputs["input_ids"][i] = sample_input_ids[-self.max_length:]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][-self.max_length:]
                labels["input_ids"][i] = label_input_ids[-self.max_length:]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def cut_text(self, examples, max_len=350):
        for i in range(len(examples)):
            if examples[i] is not None:
                max_length_text = len(examples[i].split(' '))
                if max_length_text > max_len:
                    examples[i] = ' '.join(examples[i].split(' ')[:max_len])
        return examples

    # def test_preprocess_function(self, examples):
    #     batch_size = len(examples[self.text_column])
    #     inputs = [f"Document: {x} \n Relevant Query: " for x in examples[self.text_column]]
    #     model_inputs = self.tokenizer(inputs)
    #     # print(model_inputs)
    #     for i in range(batch_size):
    #         sample_input_ids = model_inputs["input_ids"][i]
    #         model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
    #             self.max_length - len(sample_input_ids)
    #         ) + sample_input_ids
    #         model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
    #             "attention_mask"
    #         ][i]
    #         model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
    #         model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
    #     return model_inputs
    
    def get_dataset(self):
        processed_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset['train'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        # eval_dataset = self.dataset["test"].map(
        #     self.test_preprocess_function,
        #     batched=True,
        #     num_proc=1,
        #     remove_columns=self.dataset["test"].column_names,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on dataset",
        # )
        test_dataset = self.test_dataset["test"].map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        return processed_datasets, test_dataset
