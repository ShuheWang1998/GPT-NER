import os
import json
from tqdm import tqdm

def read_results(dir_, ori_dir):
    # read prompt
    file_name = os.path.join(ori_dir, "mrc-ner.test")
    file = json.load(open(file_name, "r"))
    prompt = []
    for line in file:
        prompt.append(line["context"])
    
    # read results
    file_name = os.path.join(dir_, "results.tmp")
    file = open(file_name, "r")
    results_tmp = []
    for line in file:
        line = json.loads(line)
        end_ = 0
        while end_ < len(line):
            if line[end_:end_+3] == "END":
                break
            end_ += 1
        results_tmp.append(line[:end_])
    
    print("========= changing tuple (results) =========")
    results = []
    for example_idx in tqdm(range(len(prompt))):
        prompt_token = prompt[example_idx].strip().split()
        results_token = results_tmp[example_idx].strip().split("\n")
        
        match_tuple = []
        start_ = 0
        for ner_idx in range(len(results_token)):
            ner = results_token[ner_idx].strip()
            if ner == "None":
                continue
            ner_token = ner.split()
            while start_ < len(prompt_token):
                ner_token_idx = 0
                while ner_token_idx < len(ner_token) and ner_token[ner_token_idx].strip() == prompt_token[start_+ner_token_idx].strip():
                    ner_token_idx += 1
                if ner_token_idx == len(ner_token):
                    break
                start_ += 1
            match_tuple.append((start_, start_+len(ner_token)-1))
        
        # print(prompt_token)
        # print(results_token)
        # print(match_tuple)
        results.append(match_tuple)
    return results

def read_ference(dir_):
    file_name = os.path.join(dir_, "mrc-ner.test")
    mrc_file = json.load(open(file_name, "r"))
    mrc_tuple = []

    print("========= changing tuple (reference) =========")
    for example_idx in tqdm(range(len(mrc_file))):
        mrc_ = mrc_file[example_idx]
        match_tuple = [(mrc_["start_position"][idx_], mrc_["end_position"][idx_]) for idx_ in range(len(mrc_["start_position"]))]
        mrc_tuple.append(match_tuple)
    
    return mrc_tuple

def count_f1_score(results, references):
    print("=========== computing f1 scorre ===========")

    true_positive = 0
    false_positive = 0
    false_negitative = 0

    for example_idx in tqdm(range(len(references))):
        result = results[example_idx]
        reference = references[example_idx]

        for ner in result:
            if ner in reference:
                true_positive += 1
                reference.remove(ner)
            else:
                false_positive += 1
        false_negitative += len(reference)
    
    recall = true_positive / (true_positive + false_negitative)
    precision = true_positive / (true_positive + false_positive)
    f1 = precision * recall * 2 / (recall + precision)

    return recall, precision, f1

def test():
    #file = open("/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll/results.tmp", "r")
    #for line in file:
    #    line = json.loads(line)
    #    print(line)
    #    return
    string_ = " Barbarians END-OF-TOUR"
    print(string_.strip().split("END"))

if __name__ == '__main__':
    #test()

    results_file = read_results("/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll", "/nfs1/shuhe/gpt3-ner/origin_data/conll03_mrc")
    test_file = read_ference("/nfs1/shuhe/gpt3-ner/origin_data/conll03_mrc")
    print(count_f1_score(results_file, test_file))