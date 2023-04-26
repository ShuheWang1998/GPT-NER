import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def read_mrc(file_name):
    print(f"read ... {file_name}")

    return json.load(open(file_name))

def read_word_feature(dir_, prefix, np_float=np.float16):
    print(f"read ... {dir_}, {prefix}")

    info_file = json.load(open(os.path.join(dir_, f"{prefix}.word_feature_info.json"), "r"))
    token_num = info_file["token_sum"]
    hidden_size = info_file["hidden_size"]
    
    index_list = []
    index_file = open(os.path.join(dir_, f"{prefix}.word_feature_index.json"))
    for line in index_file:
        index_list.append(json.loads(line.strip()))
    index_file.close()

    features = np.memmap(os.path.join(dir_, f"{prefix}.word_feature.npy"), 
                         dtype=np_float,
                         mode="r",
                         shape=(token_num, hidden_size))
    
    word_feature = {
        "info_file": info_file,
        "index_list": index_list,
        "features": torch.from_numpy(features)
    }
    return word_feature

def read_sentence_feature(dir_, prefix, np_float=np.float16):
    print(f"read ... {dir_}, {prefix}")

    info_file = json.load(open(os.path.join(dir_, f"{prefix}.sentence_feature_info.json"), "r"))
    sentence_num = info_file["sentence_num"]
    max_seq_len = info_file["max_seq_len"]
    hidden_size = info_file["hidden_size"]

    start_mask = np.memmap(os.path.join(dir_, f"{prefix}.sentence_start_mask.npy"), 
                         dtype=np.int32,
                         mode="r",
                         shape=(sentence_num, max_seq_len))

    end_mask = np.memmap(os.path.join(dir_, f"{prefix}.sentence_end_mask.npy"), 
                         dtype=np.int32,
                         mode="r",
                         shape=(sentence_num, max_seq_len))

    features = np.memmap(os.path.join(dir_, f"{prefix}.sentence_feature.npy"), 
                         dtype=np_float,
                         mode="r",
                         shape=(sentence_num, max_seq_len, hidden_size))
    
    sentence_feature = {
        "info_file": info_file,
        "start_mask": torch.from_numpy(start_mask),
        "end_mask": torch.from_numpy(end_mask),
        "features": torch.from_numpy(features).to(torch.float).cuda()
    }
    return sentence_feature

def read_gpt3_results(file_name):
    print(f"read ... {file_name}")

    results = []
    file = open(file_name, "r")
    for line in file:
        results.append(line.strip())
    file.close()
    return results

def find_knn(mrc_training_set, training_word_features, gpt3_results, test_sentence_features, knn_num=32):
    print("finding knn ...")

    training_sentence = []
    for item in mrc_training_set:
        training_sentence.append((item["context"].strip(), item["entity_label"].strip()))
    
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
    
    def extract_training_sentence(index_line):
        sentence_idx, word_start, word_len = index_line
        sentence = training_sentence[sentence_idx][0]
        word_list = sentence.strip().split()
        word = " ".join(word_list[word_start:word_start+word_len])
        entity_label = training_sentence[sentence_idx][1]

        real_entity_list = {}
        for idx_ in range(len(mrc_training_set[sentence_idx]["start_position"])):
            start_ = mrc_training_set[sentence_idx]["start_position"][idx_]
            end_ = mrc_training_set[sentence_idx]["end_position"][idx_]
            real_entity_list[" ".join(word_list[start_:end_+1])] = start_
        
        flag = False
        if word in real_entity_list and real_entity_list[word] == word_start:
            flag = True

        return sentence, word, entity_label, flag

    knn_results = []
    for item_idx in tqdm(range(len(gpt3_results))):
        entity_list = get_words(gpt3_results[item_idx].strip())
        now_test_sentence_start = test_sentence_features["start_mask"][item_idx]
        now_test_sentence_end = test_sentence_features["end_mask"][item_idx]
        now_test_feature = test_sentence_features["features"][item_idx]
        
        start_seq_len = now_test_sentence_start.shape[0]
        end_seq_len = now_test_sentence_end.shape[0]
        
        start_idx = 0
        word_index = 0
        for idx_, entity in enumerate(entity_list):
            entity_start = entity[1]
            entity_len = len(entity[0].strip().split())

            embedding = None
            while start_idx < start_seq_len:
                if now_test_sentence_start[start_idx] == 0:
                    start_idx += 1
                    continue
                if word_index == entity_start:
                    end_idx = start_idx
                    real_word_num = 0
                    while end_idx < end_seq_len:
                        if now_test_sentence_end[end_idx] == 1:
                            real_word_num += 1
                        if real_word_num == entity_len:
                            break
                        end_idx += 1
                    if real_word_num == entity_len:
                        embedding = torch.mean(now_test_feature[start_idx:end_idx+1], dim=0).view(1, -1)
                        break
                start_idx += 1
                word_index += 1
            
            sub_results = []
            if embedding is not None:
                # token_num
                cosine_similarity = []
                batch_size = 8192
                start_sim = 0
                while start_sim < training_word_features["info_file"]["token_sum"]:
                    end_sim = min(start_sim+batch_size, training_word_features["info_file"]["token_sum"])
                    cosine_similarity.append(nn.functional.cosine_similarity(embedding.expand(end_sim-start_sim, -1), training_word_features["features"][start_sim:end_sim].cuda()))
                    start_sim = end_sim
                cosine_similarity = torch.cat(cosine_similarity, dim=-1).view(-1)
                top_k_value, top_k_index = torch.topk(cosine_similarity, k=knn_num, dim=-1) # search_k
                
                top_k_index = top_k_index.view(-1).cpu().numpy()
                for knn_idx in range(knn_num):
                    index_line = training_word_features["index_list"][top_k_index[knn_idx]]
                    extracted_sentence, extracted_word, extracted_label, extracted_flag = extract_training_sentence(index_line=index_line)

                    sub_results.append((extracted_sentence, extracted_word, extracted_label, extracted_flag))
            knn_results.append(sub_results)
    
    return knn_results

def write_file(file_name, data):
    file = open(file_name, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False).strip()+'\n')
    file.close()

if __name__ == '__main__':
    # mrc_training = read_mrc(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/mrc-ner.train.dev")
    # training_word_features = read_word_feature(dir_="/nfs1/shuhe/gpt3-ner/features/conll03", prefix="train.dev", np_float=np.float16)
    # gpt3_results = read_gpt3_results(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/text-3/openai.17.knn.train.dev.sequence.fullprompt")
    # test_sentence_features = read_sentence_feature(dir_="/nfs1/shuhe/gpt3-ner/features/conll03", prefix="test.100", np_float=np.float16)
    # knn_ = find_knn(mrc_training_set=mrc_training, training_word_features=training_word_features, gpt3_results=gpt3_results, test_sentence_features=test_sentence_features, knn_num=32)
    # write_file(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/test.100.verify.knn.jsonl", data=knn_)

    mrc_training = read_mrc(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/mrc-ner.train.dev")
    training_word_features = read_word_feature(dir_="/nfs1/shuhe/gpt3-ner/features/conll03", prefix="train.dev", np_float=np.float16)
    gpt3_results = read_gpt3_results(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/text-full/openai.15.knn.train.dev.sequence.fullprompt")
    test_sentence_features = read_sentence_feature(dir_="/nfs1/shuhe/gpt3-ner/features/conll03", prefix="test", np_float=np.float16)
    knn_ = find_knn(mrc_training_set=mrc_training, training_word_features=training_word_features, gpt3_results=gpt3_results, test_sentence_features=test_sentence_features, knn_num=32)
    write_file(file_name="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/test.verify.knn.jsonl", data=knn_)