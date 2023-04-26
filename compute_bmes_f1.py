import os
import json
from tqdm import tqdm

def read_results(dir_):
    file_name = os.path.join(dir_, "test.results")
    file = open(file_name, "r")
    results = []
    for line in file:
        results = results + json.loads(line)
    return results

def read_test(dir_):
    file_name = os.path.join(dir_, "test.jsonl")
    file = open(file_name, "r")
    results = []
    for line in file:
        completion = json.loads(line)["completion"].split(" END")[0].strip().split('\n')
        results.append(completion)
    return results

def count_f1_score(results, references):
    print("=========== computing f1 scorre ===========")

    true_positive = 0
    false_positive = 0
    false_negitative = 0

    for example_idx in tqdm(range(len(references))):
        result = results[example_idx]
        reference = references[example_idx]
        
        for token_idx in range(min(len(reference), len(result))):
            sub_result = result[token_idx].strip()
            sub_reference = reference[token_idx].strip()
            
            if sub_reference == sub_result and sub_reference != "None":
                true_positive += 1
            else:
                if sub_reference != "None":
                    false_negitative += 1
                if sub_result != "None":
                    false_positive += 1
        if len(result) < len(reference):
            for token_idx in range(len(result), len(reference)):
                sub_reference = reference[token_idx].strip()
                if sub_reference != "None":
                    false_negitative += 1
    
    recall = true_positive / (true_positive + false_negitative)
    precision = true_positive / (true_positive + false_positive)
    f1 = precision * recall * 2 / (recall + precision)

    return recall, precision, f1

if __name__ == '__main__':
    resulst_file = read_results("/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_bert")
    test_file = read_test("/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_bert")
    print(count_f1_score(resulst_file, test_file))