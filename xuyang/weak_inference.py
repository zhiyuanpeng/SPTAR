from transformers import AutoModelForCausalLM
import torch
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import csv
from peft import PeftModel, PeftConfig
import argparse
import pandas as pd
from random import randint
from default_prompt import DefaultPrompt

# fixed_one_shot_prompt = "Document: Emphysema, also called pulmonary emphysema , condition characterized by widespread destruction of the gas-exchanging tissues of the lungs, resulting in abnormally large air spaces. Lungs affected by emphysema show loss of alveolar walls and destruction of alveolar capillaries. As a result, the surface available for the exchange of oxygen and carbon dioxide between inhaled air and blood traversing the lungs is reduced. \n Relevant Query: what effect does emphysema have on the lungs"
# fixed_two_shot_prompt = "Document: Emphysema, also called pulmonary emphysema , condition characterized by widespread destruction of the gas-exchanging tissues of the lungs, resulting in abnormally large air spaces. Lungs affected by emphysema show loss of alveolar walls and destruction of alveolar capillaries. As a result, the surface available for the exchange of oxygen and carbon dioxide between inhaled air and blood traversing the lungs is reduced. \n Relevant Query: what effect does emphysema have on the lungs \n Document: Actually your cervix will close and become low right after the egg drops due to the surge of hormones. If your still open I'd BD. Good luck! Your cervix can actually remain open for a week AFTER you release your egg. usually it will close 24 hours after, but that is nto always the case. \n Relevant Query: how long is cervix open after ovulation"

# fixed_one_shot_prompt = "Document: The bottom half of a muffaletta sandwich is slathered with olive salad. Sandwich made with ciabatta bread, ham, tomatoes, lettuce, and Swiss cheese. For a fancier grilled cheese, consider additions such as pears. Roast beef is a popular meat used for sandwiches. Salami is a popular ingredient in a sandwich. Honey and peanut butter sandwiches are a favorite for kids and adults alike. \n Relevant Query: different kinds of sandwich meats"
# fixed_two_shot_prompt = "Document: The bottom half of a muffaletta sandwich is slathered with olive salad. Sandwich made with ciabatta bread, ham, tomatoes, lettuce, and Swiss cheese. For a fancier grilled cheese, consider additions such as pears. Roast beef is a popular meat used for sandwiches. Salami is a popular ingredient in a sandwich. Honey and peanut butter sandwiches are a favorite for kids and adults alike. \n Relevant Query: different kinds of sandwich meats \n Document: The function of a motor neuron is to carry an electrical signal to a muscle, triggering it to either contract or relax. \n Relevant Query: what is the structure and function of a motor neuron?"

# fixed_one_shot_prompt = "Document: US currency doesn't expire, it is always legal tender.  I can see some trouble if you tried to spend a $10,000 bill (you'd be foolish to do so, since they are worth considerably more). Maybe some stores raise eyebrows at old-style $100's (many stores don't take $100 bills at all), but you could swap them for new style at a bank if having trouble with a particular store. Old-series currency can be an issue when trying to exchange US bills in other countries, just because it doesn't expire here, doesn't mean you can't run into issues elsewhere. Other countries have different policies, for example, over the last year the UK phased in a new five pound note, and as of last month (5/5/2017) the old fiver is no longer considered legal tender (can still swap out old fivers at the bank for now at least).  Edit: I mistook which currency you took where, and focused on US currency instead of Canadian, but it looks like it's the same story there. \n Relevant Query: What is the lifespan of a series of currency?"
# fixed_two_shot_prompt = "Document: US currency doesn't expire, it is always legal tender.  I can see some trouble if you tried to spend a $10,000 bill (you'd be foolish to do so, since they are worth considerably more). Maybe some stores raise eyebrows at old-style $100's (many stores don't take $100 bills at all), but you could swap them for new style at a bank if having trouble with a particular store. Old-series currency can be an issue when trying to exchange US bills in other countries, just because it doesn't expire here, doesn't mean you can't run into issues elsewhere. Other countries have different policies, for example, over the last year the UK phased in a new five pound note, and as of last month (5/5/2017) the old fiver is no longer considered legal tender (can still swap out old fivers at the bank for now at least).  Edit: I mistook which currency you took where, and focused on US currency instead of Canadian, but it looks like it's the same story there. \n Relevant Query: What is the lifespan of a series of currency? \n Document: An article linked from cnn.com has some great advice, which I think are good rules of thumb. Also, at least my insurance gives a premium price for those who haven't filed a claim in 5 or more years for homeowners or rental insurance. See if you have a similar discount, will loose it, and guess how much that will cost you over 5 years. My rule of thumb: Your premium might go up quite a bit, possibly as much as triple, especially for a large claim. But, it is certainly worth it if you are going to get more than triple your premium through your claim. The worst case: Mortgage mandated insurance, which will be about triple your current cost. \n Relevant Query: At what point does it become worth it to file an insurance claim?"

# fixed_one_shot_prompt = "Document: Gnarkill was an American parody band that formed in West Chester, Pennsylvania in 2002 and comprised vocalist Brandon DiCamillo, guitarist Rich Vose, keyboardist Bam Margera, drummer Jess Margera and mixer Matt Cole. The band released two albums: ""Gnarkill"" (2002) and ""Gnarkill vs. Unkle Matt and the Shitbirdz"" (2006). \n Relevant Query: For which crew was the vocalist of Gnarkill a founding member?"
# fixed_two_shot_prompt = "Document: Gnarkill was an American parody band that formed in West Chester, Pennsylvania in 2002 and comprised vocalist Brandon DiCamillo, guitarist Rich Vose, keyboardist Bam Margera, drummer Jess Margera and mixer Matt Cole. The band released two albums: ""Gnarkill"" (2002) and ""Gnarkill vs. Unkle Matt and the Shitbirdz"" (2006). \n Relevant Query: For which crew was the vocalist of Gnarkill a founding member? \n Document: Khrunichev T-411 Aist. The Krunichev T-411 Aist (en: ""Stork"") is a Russian light utility monoplane designed by the Russian company Aeroprogress and placed into production by the Khrunichev State Research and Production Space Center. A version is marketed in the United States as the Aeroprogress T-411 Wolverine powered by a Continental TSIO-550-B. \n Relevant Query: How many cylinders does the engine in an Aeroprogress T-411 Wolverine have?"

# fixed_one_shot_prompt = "Document: Robert Arum ( born December 8 , 1931 ) is an American lawyer , boxing promoter and businessman . He is the founder and CEO of Top Rank , a professional boxing promotion company based in Las Vegas . He also worked for the US Attorney 's Office for the southern district of New York in the tax division during his legal career before moving into boxing promotion . \n Relevant Query: Bob Arum was incapable of being in the legal career."
# fixed_two_shot_prompt = "Document: Robert Arum ( born December 8 , 1931 ) is an American lawyer , boxing promoter and businessman . He is the founder and CEO of Top Rank , a professional boxing promotion company based in Las Vegas . He also worked for the US Attorney 's Office for the southern district of New York in the tax division during his legal career before moving into boxing promotion . \n Relevant Query: Bob Arum was incapable of being in the legal career. \n Document: Iain Glen ( born 24 June 1961 ) is a Scottish film , television , and stage actor . Glen is best known for his roles as Dr. Alexander Isaacs / Tyrant in the Resident Evil films and for portraying Ser Jorah Mormont on Game of Thrones . Other notable roles include John Hanning Speke in Mountains of the Moon , Sir Richard Carlisle in Downton Abbey , the title role in Jack Taylor and Jarrod Slade in Cleverman . \n Relevant Query: Iain Glen is a fish."

def simple_filter(text):
    text = text.split("\n")[0]
    for punt in [".", ",", "?"]:
        pre_i = ""
        new_text = ""
        for i in text.split(punt):
            if pre_i != i:
                new_text += i
                pre_i = i
            else:
                break
        text = new_text
    return text

def fix_prompt_controller(dataset_name):
    if 'ms_' in dataset_name:
        fixed_one_shot_prompt = DefaultPrompt.ms_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.ms_50_fixed_two_shot_prompt
    if dataset_name == 'fiqa_50':
        fixed_one_shot_prompt = DefaultPrompt.fiqa_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.fiqa_50_fixed_two_shot_prompt
    if dataset_name == 'hotpotqa_50':
        fixed_one_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt
    if dataset_name == 'fever_50':
        fixed_one_shot_prompt = DefaultPrompt.fever_50_fixed_one_shot_prompt
        fixed_two_shot_prompt = DefaultPrompt.fever_50_fixed_two_shot_prompt
    return fixed_one_shot_prompt, fixed_two_shot_prompt

def weak_infer(model, tokenizer, corpus_filtered, device, weak_queries_path, weak_train_path, prompt_num, train_prompt, dataset_name, new_query_id=2000000, fixed_prompt=True):
    model.eval()
    weak_query_list = []
    weak_query_doc_id_list = []
    train_prompt_len = len(train_prompt)
    if fixed_prompt:
        fixed_one_shot_prompt, fixed_two_shot_prompt = fix_prompt_controller(dataset_name)
    for i in tqdm(corpus_filtered):
        # try:
        new_query_id += 1
        corpus_id = i['_id']
        corpus_text = i['text']
        gen_trail = 0
        while(gen_trail < 3):
            if int(prompt_num) == 2:
                if fixed_prompt:
                    new_prompt = "{} \n Document: {} \n Relevant Query: ".format(fixed_one_shot_prompt, corpus_text)
                else:
                    prompt_idx_1 = randint(0, train_prompt_len-1)
                    new_prompt = "Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".format(train_prompt[prompt_idx_1]['text_x'], train_prompt[prompt_idx_1]['text_y'], corpus_text)
            elif int(prompt_num) == 3:
                if fixed_prompt:
                    new_prompt = "{} \n Document: {} \n Relevant Query: ".format(fixed_two_shot_prompt, corpus_text)
                else:
                    prompt_idx_1 = randint(0, train_prompt_len-1)
                    prompt_idx_2 = randint(0, train_prompt_len-1)
                    new_prompt = "Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: {} \n Document: {} \n Relevant Query: ".format(train_prompt[prompt_idx_1]['text_x'], train_prompt[prompt_idx_1]['text_y'], train_prompt[prompt_idx_2]['text_x'], train_prompt[prompt_idx_2]['text_y'], corpus_text)
            else:
                new_prompt = "Document: {} \n Relevant Query: ".format(corpus_text)
            inputs = tokenizer(new_prompt, return_tensors="pt")
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    max_new_tokens=128, 
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.7
                )
                new_query = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(new_prompt):]
                new_query = simple_filter(new_query)
            if len(new_query) == 0:
                gen_trail += 1
            else:
                gen_trail = 4
        weak_query_doc_id_list.append([new_query_id, corpus_id, 1])
        weak_query_list.append({"_id": str(new_query_id), "text": new_query, "metadata": {}})
        # except:
        #     pass
    with open(weak_queries_path, 'w') as outfile:
        for entry in weak_query_list:
            json.dump(entry, outfile)
            outfile.write('\n')
    with open(weak_train_path, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(["query-id", "corpus-id", "score"])
        for qd_i in weak_query_doc_id_list:
            tsv_output.writerow(qd_i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu device")
    parser.add_argument("--weak_queries_path", type=str, default="", help="tokenizer")
    parser.add_argument("--weak_train_path", type=str, default="", help="tokenizer")
    parser.add_argument("--peft_model_id", type=str, help="our model checkpoint")
    parser.add_argument("--data_path", type=str, help="data")
    parser.add_argument("--prompt_num", type=int, default=1, help="data")
    parser.add_argument("--train_prompt_path", type=str, help="train_prompt_path")
    parser.add_argument("--dataset_name", type=str, default="ms_50", help="dataset_name")
    parser.add_argument("--new_query_id", type=int, default=2000000, help="dataset_name")
    args = parser.parse_args()

    config = PeftConfig.from_pretrained(args.peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to(args.device)
    corpus_filtered = json.loads(pd.read_csv(args.data_path).to_json(orient="records"))
    train_prompt = json.loads(pd.read_csv(args.train_prompt_path).to_json(orient="records"))
    weak_infer(model, tokenizer, corpus_filtered, args.device, args.weak_queries_path, args.weak_train_path, args.prompt_num, train_prompt, args.dataset_name, int(args.new_query_id))