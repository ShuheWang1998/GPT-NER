import json
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate-file", type=str, help="file for the predictions")
    parser.add_argument("--reference-file", type=str, help="file for the reference")
    
    return parser

def read_openai_file(file_name):
    print(f"read ... {file_name}")

    file = open(file_name, "r")
    results = []
    for line in tqdm(file):
        results.append(line.strip())
    file.close()
    return results

def read_mrc_file(file_name):
    print(f"read ... {file_name}")

    return json.load(open(file_name))

def compute_f1(mrc_data, openai_data):
    print("computting f1 ...")

    true_positive = 0
    false_positive = 0
    false_negitative = 0
    for idx_ in range(len(mrc_data)):
        reference = []
        candidate = []
        item_ = mrc_data[idx_]
        context_list = item_["context"].strip().split()
        for sub_idx in range(len(item_["start_position"])):
            start_ = item_["start_position"][sub_idx]
            end_ = item_["end_position"][sub_idx]
            reference.append((" ".join(context_list[start_:end_+1]), start_, end_))
        

        flag = False
        candidate_sentence = openai_data[idx_]
        candidate_sentence_list = candidate_sentence.strip().split()
        start_ = 0
        for word_idx, word in enumerate(candidate_sentence_list):
            if len(word) > 2 and word[0] == '@' and word[1] == '@':
                flag = True
                for end_ in range(word_idx, len(candidate_sentence_list)):
                    end_word = candidate_sentence_list[end_]
                    if len(end_word) > 2 and end_word[-1] == '#' and end_word[-2] == '#':
                        entity_ = " ".join(candidate_sentence_list[word_idx:end_+1])[2:-2]
                        len_ = end_ - word_idx + 1
                        while start_ < len(context_list):
                            if start_ + len_ - 1 < len(context_list) and " ".join(context_list[start_:start_+len_]) == entity_:
                                candidate.append((" ".join(context_list[start_:start_+len_]), start_, start_ + len_ - 1))
                                break
                            start_ += 1
                        break
            if len(word) > 2 and word[-1] == '#' and word[-2] == '#':
                flag = False
                continue
            if not flag:
                start_ += 1

        # item_ = openai_data[idx_]
        # context_list = item_.strip().split()

        # flag = False
        # start_ = 0
        # for word_idx, word in enumerate(context_list):
        #     if len(word) > 2 and word[0] == '@' and word[1] == '@':
        #         flag = True
        #         start_ = word_idx
        #     if flag and len(word) > 2 and word[-1] == '#' and word[-2] == '#':
        #         flag = False
        #         candidate.append((" ".join(context_list[start_:word_idx+1])[2:-2], start_, word_idx))
        
        # print(f"ref: {reference}")
        # print(f"can: {candidate}")
        for span_item in candidate:
            if span_item in reference:
                reference.remove(span_item)
                true_positive += 1
            else:
                false_positive += 1
        false_negitative += len(reference)
    
    span_recall = true_positive / (true_positive + false_negitative)
    span_precision = true_positive / (true_positive + false_positive)
    span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision)

    return span_recall, span_precision, span_f1

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    predictions = read_openai_file(args.candidate_file)
    mrc_data = read_mrc_file(args.reference_file)

    span_recall, span_precision, span_f1 = compute_f1(mrc_data=mrc_data, openai_data=predictions)

    print(f"span_recall: {span_recall}, span_precision: {span_precision}, span_f1: {span_f1}")