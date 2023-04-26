from simcse import SimCSE
import json
import numpy as np
import os
import faiss
import random


def read_feature(dir_, prefix):
    info_file = json.load(open(os.path.join(dir_, f"{prefix}.start_word_feature_info.json")))
    features = np.memmap(os.path.join(dir_, f"{prefix}.start_word_feature.npy"), 
                         dtype=np.float32,
                         mode="r",
                         shape=(info_file["entity_num"], info_file["hidden_size"]))
    index_file = []
    file = open(os.path.join(dir_, f"{prefix}.start_word_feature_index.json"), "r")
    for line in file:
        index_file.append(int(line.strip()))
    file.close()
    return info_file, features, index_file

def read_mrc_data(dir_, prefix):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(dir_, prefix="test"):
    print("reading ...")
    file_name = os.path.join(dir_, f"{prefix}.knn.jsonl")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx

def compute_mrc_knn(test_info, test_features, train_info, train_features, train_index, knn_num=32):
    quantizer = faiss.IndexFlatIP(train_info["hidden_size"])
    index = quantizer
    index.add(train_features.astype(np.float32))
    # 10 is a default setting in simcse
    index.nprobe = min(10, train_info["entity_num"])
    index = faiss.index_gpu_to_cpu(index)

    top_value, top_index = index.search(test_features.astype(np.float32), knn_num)

    sum_ = 0
    vis_index = {}
    for idx_, value in enumerate(train_index):
        if value == 0:
            continue
        for i in range(sum_, value+sum_):
            vis_index[i] = idx_
        sum_ += value

    example_idx = [[vis_index[int(i)] for i in top_index[idx_]] for idx_ in range(test_info["entity_num"])]
    example_value = [[float(value) for value in top_value[idx_]] for idx_ in range(test_info["entity_num"])]

    return example_idx, example_value

def compute_simcse_knn(test_mrc_data, train_mrc_data, knn_num, test_index=None):
    sim_model = SimCSE("/data2/wangshuhe/gpt3_ner/models/sup-simcse-roberta-large")

    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item["entity_label"]
        context = item["context"]
        # if len(item["start_position"]) == 0:
        #     if label not in train_sentence:
        #         train_sentence[label] = []
        #         train_sentence_index[label] = []
        #     train_sentence[label].append(context)
        #     train_sentence_index[label].append(idx_)
        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)
    
    train_index = {}
    for key, _ in train_sentence.items():
        embeddings = sim_model.encode(train_sentence[key], batch_size=128, normalize_to_unit=True, return_numpy=True)
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = quantizer
        index.add(embeddings.astype(np.float32))
        # 10 is a default setting in simcse
        index.nprobe = min(10, len(train_sentence[key]))
        index = faiss.index_gpu_to_cpu(index)

        train_index[key] = index

    example_idx = []
    example_value = []

    if test_index is None:
        for idx_ in range(len(test_mrc_data)):
            context = test_mrc_data[idx_]["context"]
            label = test_mrc_data[idx_]["entity_label"]

            embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
            top_value, top_index = train_index[label].search(embedding.astype(np.float32), knn_num)

            example_idx.append([train_sentence_index[label][int(i)] for i in top_index[0]])
            example_value.append([float(value) for value in top_value[0]])
        
        return example_idx, example_value

    for idx_, sub_index in enumerate(test_index):
        if sub_index != 0:
            continue
        context = test_mrc_data[idx_]["context"]
        label = test_mrc_data[idx_]["entity_label"]

        embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
        top_value, top_index = train_index[label].search(embedding.astype(np.float32), knn_num)

        example_idx.append([train_sentence_index[label][int(i)] for i in top_index[0]])
        example_value.append([float(value) for value in top_value[0]])
    
    return example_idx, example_value

def combine_full_knn(test_index, mrc_knn_index, simcse_knn_index):
    results = []
    mrc_idx = 0
    simcse_idx = 0
    for idx_, num in enumerate(test_index):
        if num == 0:
            results.append(simcse_knn_index[simcse_idx])
            simcse_idx += 1
        else:
            knn_num = len(mrc_knn_index[mrc_idx])
            span_ = int(knn_num // num)
            if span_ * num != knn_num:
                span_ += 1
            sub_results = []
            for sub_idx in range(mrc_idx, mrc_idx+num):
                sub_results = sub_results + mrc_knn_index[sub_idx][:span_]
            sub_results = sub_results[:knn_num]
            results.append(sub_results)
            mrc_idx += num
    
    return results

def random_knn(test_mrc_data, train_mrc_data, knn_num):
    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item["entity_label"]
        context = item["context"]

        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)

    example_idx = []

    for idx_ in range(len(test_mrc_data)):
        context = test_mrc_data[idx_]["context"]
        label = test_mrc_data[idx_]["entity_label"]

        random.shuffle(train_sentence_index[label])

        example_idx.append(train_sentence_index[label][:knn_num])
    
    return example_idx, None

def write_file(dir_, data):
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False)+'\n')
    file.close()

if __name__ == '__main__':
    # test_info, test_features, test_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding", prefix="test.100")
    # test_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="test")
    # train_info, train_features, train_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding", prefix="train.dev")
    # train_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="train.dev")
    # # test_info, test_features, test_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding_sorted", prefix="test")
    # # test_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="test")
    # # train_info, train_features, train_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding_sorted", prefix="train.dev.sorted")
    # # train_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="train.dev.sorted")

    # mrc_knn_idx, mrc_knn_value = compute_mrc_knn(test_info=test_info, test_features=test_features, train_info=train_info, train_features=train_features, train_index=train_index, knn_num=32)
    # simcse_knn_idx, simcse_knn_value = compute_simcse_knn(test_index=test_index, test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)

    # combined_data = combine_full_knn(test_index=test_index, mrc_knn_index=mrc_knn_idx, simcse_knn_index=simcse_knn_idx)

    # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding/test.100.full.knn.jsonl", data=combined_data)
    # # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/start_word_embedding_sorted/test.full.knn.jsonl", data=combined_data)


    # test_info, test_features, test_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/start_word_embedding", prefix="test")
    # test_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/", prefix="test")
    # train_info, train_features, train_index = read_feature(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/start_word_embedding", prefix="train.dev")
    # train_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/", prefix="train.dev")

    # mrc_knn_idx, mrc_knn_value = compute_mrc_knn(test_info=test_info, test_features=test_features, train_info=train_info, train_features=train_features, train_index=train_index, knn_num=32)
    # text2vec_knn_idx = read_idx(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4", prefix="test.embedding")

    # combined_data = combine_full_knn(test_index=test_index, mrc_knn_index=mrc_knn_idx, simcse_knn_index=text2vec_knn_idx)

    # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/start_word_embedding/test.mrc.knn.jsonl", data=combined_data)

    # test_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/low_resource", prefix="test")
    # train_mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/low_resource", prefix="train.1")
    # index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/low_resource/low_resource_1_knn/test.simcse.knn.jsonl", data=index_)

    # test_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="test")
    # train_mrc_data = read_mrc_data(dir_="/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en", prefix="train.8")
    # index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    # write_file(dir_="/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en/test.8.embedding.knn.jsonl", data=index_)

    # test_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="test.100")
    # train_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="train.dev")
    # index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    # write_file(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/test.100.simcse.32.knn.jsonl", data=index_)

    # test_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/low_resource", prefix="test")
    # train_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/low_resource", prefix="train.10000")
    # index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    # write_file(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/low_resource/test.10000.simcse.32.knn.jsonl", data=index_)

    # test_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="test")
    # train_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="train.dev")
    # index_, value_ = random_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    # write_file(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/test.random.32.knn.jsonl", data=index_)

    test_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/ontonotes5_mrc", prefix="test.100")
    train_mrc_data = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/ontonotes5_mrc", prefix="dev")
    index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, knn_num=32)
    write_file(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/ontonotes5_mrc/test.100.simcse.dev.32.knn.jsonl", data=index_)