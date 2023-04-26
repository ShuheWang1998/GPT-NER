import os
import json
from text2vec import Similarity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import Tensor


def read_mrc_data(dir_, prefix):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def compute_score_text2vec(test_mrcdata, train_mrcdata, knn=32):

    def takeFirst(elem):
        return elem[0]

    ner_type_num = 0
    ner_type = {}
    for item in test_mrcdata:
        type_ = item["entity_label"]
        if type_ not in ner_type:
            ner_type[type_] = True
            ner_type_num += 1

    train_sentence = []
    for idx_, item in enumerate(train_mrcdata):
        if idx_ % ner_type_num == 0:
            train_sentence.append(item["context"])

    test_sentence = []
    for idx_, item in enumerate(test_mrcdata):
        if idx_ % ner_type_num == 0:
            test_sentence.append(item["context"])

    sim_model = Similarity("/nfs/shuhe/gpt3-ner/models/text2vec-base-chinese")
    test_embedding = sim_model.model.encode(test_sentence)
    train_embedding = sim_model.model.encode(train_sentence)
    print(type(test_embedding))

    def get_scores(sen_embedding):
        score = cos_sim(sen_embedding, train_embedding).numpy().tolist()[0]
        return score

    scores = []
    with ThreadPoolExecutor(max_workers=32) as exe:
        pbar = tqdm(total=len(test_sentence))
        for full_score in exe.map(get_scores, test_embedding):
            sub_score = [(score, idx_*ner_type_num) for idx_, score in enumerate(full_score)]
            sub_score.sort(key=takeFirst, reverse=True)
            sub_score = sub_score[:min(knn, len(train_mrcdata))]

            for span_ in range(ner_type_num):
                scores.append([(last_score, last_index+span_) for last_score, last_index in sub_score])
            
            pbar.update(1)
    
    # scores = []
    # pbar = tqdm(total=len(test_mrcdata))
    # for item_idx, item in enumerate(test_mrcdata):
    #     sentence = item["context"]
    #     sub_score = []
    #     if item_idx % ner_type_num == 0:
    #         full_score = sim_model.get_scores(sentence, train_sentence).tolist()
    #         sub_score = [(score, idx_*ner_type_num) for score in full_score]
    #         sub_score.sort(key=takeFirst, reverse=True)
    #     else:
    #         span_ = int(item_idx % ner_type_num)
    #         sub_score = [(last_score, last_index+span_) for last_score, last_index in scores[-1]]
    #     scores.append(sub_score[:knn])
    #     pbar.update(1)
    
    values = []
    index = []
    for score in scores:
        values.append([])
        index.append([])
        for value, idx_ in score:
            values[-1].append(value)
            index[-1].append(idx_)
    
    return values, index

def write_file(dir_, data):
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False)+'\n')
    file.close()

if __name__ == '__main__':
    # test_mrc = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_msra", prefix="test")[:2]
    # train_mrc = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_msra", prefix="train.dev")
    # values, index = compute_score_text2vec(test_mrcdata=test_mrc, train_mrcdata=train_mrc, knn=32)
    # print(values)
    # print(index)
    # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_msra/test.embedding.knn.jsonl", data=index)

    # test_mrc = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4", prefix="test")
    # train_mrc = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4", prefix="train.dev")
    # values, index = compute_score_text2vec(test_mrcdata=test_mrc, train_mrcdata=train_mrc, knn=32)
    # write_file(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/zh_onto4/test.embedding.knn.jsonl", data=index)

    test_mrc = read_mrc_data(dir_="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc", prefix="test")
    train_mrc = read_mrc_data(dir_="/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en", prefix="train.8")
    values, index = compute_score_text2vec(test_mrcdata=test_mrc, train_mrcdata=train_mrc, knn=32)
    write_file(dir_="/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en/test.8.embedding.knn.jsonl", data=index)