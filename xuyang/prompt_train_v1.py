from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from args import PromptTuringArgs
from utils import AverageMeter, setup_train, get_device
from torch.utils.tensorboard import SummaryWriter
from dataset import MSMARCODataset, MSMARCOPointWiseDataset
import argparse
from utils import reset_args


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def main(args):
    # config
    export_root, args = setup_train(args)
    log_writer = SummaryWriter(export_root)
    tokenizer = load_tokenizer(args)

    # dataset
    ir_dataset = MSMARCODataset(args, tokenizer)
    # ir_dataset = MSMARCOPointWiseDataset(args, tokenizer)
    train_dataset, test_dataset = ir_dataset.get_dataset()
    train_dataloader = DataLoader(train_dataset['train'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(train_dataset['test'], collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    
    # creating model
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        tokenizer_name_or_path=args.model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # training and evaluation
    model = model.to(args.device)
    original_eval_loss = 999999
    early_stop_epoch = 0
    for epoch in range(args.num_epochs):
        if early_stop_epoch > 5:
            print('Terminating because of early stopping!')
            break
        total_train_loss = 0
        total_eval_loss = 0
        avg_train_loss = AverageMeter()
        avg_val_loss = AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.detach().float()
            avg_train_loss.update(loss.detach().float().item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # evaluate eval dataset
        
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.detach().float()
            avg_val_loss.update(loss.detach().float().item())
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
        
        # get metrics
        train_epoch_loss = total_train_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss).detach().float().item()
        eval_epoch_loss = total_eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss).detach().float().item()
        # saving model
        if avg_val_loss.avg < original_eval_loss:
            original_eval_loss = avg_val_loss.avg
            early_stop_epoch = 0
            filepath = Path(export_root).joinpath(args.peft_model_id)
            print('new best val loss, model saved')
            model.save_pretrained(filepath)
        else:
            early_stop_epoch += 1
        # logger
        log_writer.add_scalar('Training/train_loss', avg_train_loss.avg, epoch)
        log_writer.add_scalar('Training/val_loss', avg_val_loss.avg, epoch)
        log_writer.add_scalar('Training/train_ppl', train_ppl, epoch)
        log_writer.add_scalar('Training/eval_ppl', eval_ppl, epoch)
        # print('train epoch: ', epoch, ' train loss = ', "{:.5f}".format(avg_train_loss.avg), ' val loss = ', "{:.5f}".format(avg_val_loss.avg), 
        #       ' test loss = ', "{:.5f}".format(avg_test_loss.avg), ' train ppl = ', "{:.5f}".format(avg_train_loss.avg), ' train loss = ', "{:.5f}".format(avg_train_loss.avg),)
        print(f"{epoch=}: {train_ppl=} {avg_train_loss.avg=} {eval_ppl=} {avg_val_loss.avg=}")
    # evaluate test dataset
    # total_test_loss = 0
    # avg_test_loss = AverageMeter()
    # model.eval()
    # for step, batch in enumerate(tqdm(test_dataloader)):
    #     batch = {k: v.to(args.device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #     loss = outputs.loss
    #     total_test_loss += loss.detach().float()
    #     avg_test_loss.update(loss.detach().float().item())
    # test_epoch_loss = total_test_loss / len(test_dataloader)
    # test_ppl = torch.exp(test_epoch_loss).detach().float().item()
    # log_writer.add_scalar('Training/test_loss', avg_test_loss.avg, epoch)
    # log_writer.add_scalar('Training/test_ppl', test_ppl, epoch)
    # print(f"{epoch=}: {train_ppl=} {avg_train_loss.avg=} {eval_ppl=} {avg_val_loss.avg=} {test_ppl=} {avg_test_loss.avg=}")
    log_writer.close()

    
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
    parser.add_argument("--few_shot_num", type=int, help="few shot setting")
    args = parser.parse_args(namespace=base_args)
    args = reset_args(args)
    main(args)

# python prompt_train_v1.py --device_idx 1 --num_virtual_tokens 50 --prompt_num 3 --llm_name llama-7b --train_data /home/xwu/project/SPTAR/xuyang/data/msmarco_50/prompt_tuning_1000_train_text_sampled.csv --eval_data /home/xwu/project/SPTAR/xuyang/data/msmarco_50/prompt_tuning_50_test_text.csv --test_data /home/xwu/project/SPTAR/xuyang/data/msmarco_50/prompt_tuning_50_test_text.csv --few_shot_num 100 --dataset_name ms_50