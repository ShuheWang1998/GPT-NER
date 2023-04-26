import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random
from simcse import SimCSE
import numpy as np
import faiss

random.seed(1)
logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mrc-dir", type=str, help="directory for the mrc input")
    parser.add_argument("--mrc-name", type=str, help="file name for the mrc input")
    parser.add_argument("--gpt-dir", type=str, help="directory for the gpt input")
    parser.add_argument("--gpt-name", type=str, help="file name for the gpt input")
    parser.add_argument("--data-name", type=str, help="dataset name for the input")
    parser.add_argument("--knn-file", default="None", type=str, help="knn file for the input")
    parser.add_argument("--write-dir", type=str, help="directory for the output")
    parser.add_argument("--write-name", type=str, help="file name for the output")
    parser.add_argument("--knn-num", type=int, default=1, help="numebr for the knn")
    
    return parser

def read_mrc_data(dir_, prefix="test"):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_results(dir_, prefix="test"):
    file_name = os.path.join(dir_, prefix)
    file = open(file_name, "r")
    results = []
    for line in tqdm(file):
        results.append(line.strip())
    file.close()
    return results

def read_knn_file(file_name):
    file = open(file_name, "r")
    results = []
    for line in tqdm(file):
        results.append(json.loads(line.strip()))
    file.close()
    return results

def transferPrompt(mrc_data, gpt_results, data_name="CONLL", knn_results=None, knn_num=14):
    print("tansferring prompt ...")

    def get_words(labeled_sentence):
        word_list = []
        words = labeled_sentence.strip().split()
        flag = False
        last_ = ""
        for idx_, word in enumerate(words):
            if len(word) > 2 and word[0] == '@' and word[1] == '@':
                last_ = idx_
                flag = True
            if flag and len(word) > 2 and word[-1] == '#' and word[-2] == '#':
                word_list.append((" ".join(words[last_:idx_+1])[2:-2], last_))
                flag = False
        return word_list
    
    def get_knn(index_, test_label):
        if len(knn_results[index_]) == 0:
            return None
        prompt = ""
        for sentence, word, knn_label, knn_flag in knn_results[index_][:knn_num]:
            transfered_label = FULL_DATA[data_name][knn_label][0]
            answer = "Yes" if transfered_label == test_label and knn_flag else "No"

            prompt += f"The given sentence: {sentence}\nIs the word \"{word}\" in the given sentence an {test_label} entity? Please answer with yes or no.\n{answer}\n\n"
        
        return prompt

    prompts = []
    entity_index = []
    prompts_nums = []
    knn_idx = 0
    for item_idx in tqdm(range(len(mrc_data))):
        item_ = mrc_data[item_idx]
        context = item_["context"]
        origin_label = item_["entity_label"]
        transfered_label, sub_prompt = FULL_DATA[data_name][origin_label]
        upper_transfered_label = transfered_label[0].upper() + transfered_label[1:]
        entity_list = get_words(gpt_results[item_idx].strip())

        if origin_label == "PER":
            entity_list = []

        prompts_num = 0
        for entity, entity_idx in entity_list:
            # prompt = f"{context}\nIs the word \"{entity}\" in the former sentence an {transfered_label} entity? Please answer with yes or no."
            prompt = f"You are an excellent linguist. The task is to verify whether the word is an {transfered_label} entity extracted from the given sentence. {upper_transfered_label} entities {sub_prompt}.\n\n"
            if knn_results is None:
                prompt += f"The given sentence: {context}\nIs the word \"{entity}\" in the given sentence an {transfered_label} entity? Please answer with yes or no.\n"
                prompts.append(prompt)
                entity_index.append((entity_idx, len(entity.strip().split())))
                prompts_num += 1
            else:
                knn_prompt = get_knn(knn_idx, test_label=transfered_label)
                if knn_prompt != "":
                    knn_idx += 1
                    prompt += knn_prompt
                    prompt += f"The given sentence: {context}\nIs the word \"{entity}\" in the given sentence a {transfered_label} entity? Please answer with yes or no.\n"
                    prompts.append(prompt)
                    entity_index.append((entity_idx, len(entity.strip().split())))
                    prompts_num += 1
            # print(prompt)
        prompts_nums.append(prompts_num)
    return prompts, entity_index, prompts_nums

def ner_access(openai_access, prompts, batch=16):
    print("accessing ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(prompts))
    while start_ < len(prompts):
        end_ = min(start_+batch, len(prompts))
        results = results + openai_access.get_multiple_sample(prompts[start_:end_])
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()
    return results

def construct_results(gpt_results, entity_index, prompts_num, verify_results):

    def justify(string_):
        if len(string_) >= 3 and string_[:3].lower() == "yes":
            return "yes"
        if len(string_) >= 2 and string_[:2].lower() == "no":
            return "no"
        return ""

    results = []
    start_ = 0
    for idx_, item in enumerate(gpt_results):
        words_list = item.strip().split()
        now_num = prompts_num[idx_]
        for sub_idx in range(now_num):
            num = start_ + sub_idx
            if justify(verify_results[num].strip()) == "yes":
                continue
            elif justify(verify_results[num].strip()) == "no":
                start_index, len_ = entity_index[num]
                words_list[start_index] = words_list[start_index][2:]
                words_list[start_index+len_-1] = words_list[start_index+len_-1][:-2]
        start_ += now_num
        results.append(" ".join(words_list))
    return results

def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(line.strip()+'\n')
    file.close()

if __name__ == '__main__':
    # test()

    parser = get_parser()
    args = parser.parse_args()

    openai_access = AccessBase(
        engine="text-davinci-003",
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )

    mrc_test = read_mrc_data(dir_=args.mrc_dir, prefix=args.mrc_name)
    gpt_results = read_results(dir_=args.gpt_dir, prefix=args.gpt_name)

    knn_results = None
    if args.knn_file != "None":
        knn_results = read_knn_file(file_name=args.knn_file)

    prompts, entity_idx, prompts_nums = transferPrompt(mrc_data=mrc_test, gpt_results=gpt_results, data_name=args.data_name, knn_results=knn_results, knn_num=args.knn_num)
    verify_results = ner_access(openai_access=openai_access, prompts=prompts, batch=1)
    final_results = construct_results(gpt_results=gpt_results, entity_index=entity_idx, prompts_num=prompts_nums, verify_results=verify_results)

    # print(final_results)

    write_file(labels=final_results, dir_=args.write_dir, last_name=args.write_name)
    # write_file(labels=verify_results, dir_=args.write_dir, last_name="only.verify.1")
