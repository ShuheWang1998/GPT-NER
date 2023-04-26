import os
from logger import get_logger
import json
import random
from simcse import SimCSE
import numpy as np

random.seed(1)
logger = get_logger(__name__)

def read_mrc_data(dir_, prefix="test"):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def compute_feature(mrc_data, sim_path, batch_size, dir_, prefix):
    sim_model = SimCSE(sim_path)
    sentences = [item["context"] for item in mrc_data]
    
    json_info = {
        "sentence_num": len(sentences),
        "hidden_size": 1024
        }
    json.dump(json_info, open(os.path.join(dir_, f"{prefix}.simcse.feature_info.json"), "w"),
                sort_keys=True, indent=4, ensure_ascii=False)
    print(len(sentences))
    feature_file = os.path.join(dir_, f"{prefix}.simcse.sentence_feature.npy")
    features = np.memmap(feature_file, 
                         dtype=np.float32,
                         mode="w+",
                         shape=(len(sentences), 1024))
    features_in_memory = np.zeros((len(sentences), 1024), dtype=np.float32)

    start_ = 0
    while start_ < len(sentences):
        end_ = min(len(sentences), start_ + batch_size)
        embeddings = sim_model.encode(sentences[start_:end_], batch_size=end_-start_, normalize_to_unit=True, return_numpy=True)
        features_in_memory[start_:end_] = embeddings[:]
        start_ = end_
    
    features[:] = features_in_memory[:]


if __name__ == '__main__':
    # mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ontonotes5_mrc/", prefix="test")
    # compute_feature(mrc_data=mrc_data, sim_path="/nfs1/shuhe/gpt3-nmt/sup-simcse-roberta-large", batch_size=128, dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ontonotes5_mrc/", prefix="test")

    # mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ace2004/", prefix="test")
    # compute_feature(mrc_data=mrc_data, sim_path="/nfs1/shuhe/gpt3-nmt/sup-simcse-roberta-large", batch_size=128, dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ace2004/", prefix="test")

    # mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ace2005/", prefix="train.dev")
    # compute_feature(mrc_data=mrc_data, sim_path="/nfs1/shuhe/gpt3-nmt/sup-simcse-roberta-large", batch_size=128, dir_="/nfs/shuhe/gpt3-ner/gpt3-data/ace2005/", prefix="train.dev")

    # mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/genia/", prefix="test")
    # compute_feature(mrc_data=mrc_data, sim_path="/nfs1/shuhe/gpt3-nmt/sup-simcse-roberta-large", batch_size=128, dir_="/nfs/shuhe/gpt3-ner/gpt3-data/genia/", prefix="test")

    mrc_data = read_mrc_data(dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="train.dev")
    compute_feature(mrc_data=mrc_data, sim_path="/nfs1/shuhe/gpt3-nmt/sup-simcse-roberta-large", batch_size=128, dir_="/nfs/shuhe/gpt3-ner/gpt3-data/conll_mrc/", prefix="train.dev")