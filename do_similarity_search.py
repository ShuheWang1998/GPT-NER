import argparse
import os
from tqdm import tqdm
import faiss
from simcse import SimCSE
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from openai_access.dataset_name import FULL_DATA


def read_file(file_path):
    return json.load(open(file_path))

def write_file(file_path, results):
    print("============ writting ============")
    file = open(file_path, "w")
    for idx_ in tqdm(range(len(results))):
        file.write(json.dumps(results[idx_])+'\n')
    file.close()

def main():
    parser = argparse.ArgumentParser(description='build indexes for similarity search')
    parser.add_argument("--data", type=str, required=True, help="paths to sentences")
    parser.add_argument("--index", type=str, required=True, help="paths to index")
    parser.add_argument("--index-name", type=str, required=True, help="name for the extracted index")
    parser.add_argument("--batch-size", type=int, default=64, required=False, help="batch size for search")
    parser.add_argument("--threshold", type=float, default=0.0, required=False, help="only return results with cosine similarities higher than the threshold")
    parser.add_argument("--top-k", type=int, default=5, required=False, help="return top-k results")
    parser.add_argument('--use-gpu', default=False, action='store_true', help='if true, use gpu for building')
    parser.add_argument("--model-path", type=str, required=True, help="paths for simcse model")
    parser.add_argument("--output", type=str, required=True, help="paths to output")
    parser.add_argument('--not-faiss', default=False, action='store_true', help='if true, not use faiss for building')
    parser.add_argument("--datastore-file", type=str, required=False, help="paths to original datastore file")

    args = parser.parse_args()

    test_file = read_file(args.data)
    sentences = [item_["context"] for item_ in test_file]
    test_name = args.data_name

    sim_model = SimCSE(args.model_path)
    
    print("============ computing embeddings ============")
    device = "cuda" if args.use_gpu else "cpu"
    
    if not os.path.exists(os.path.join(args.index, "query.embeddings.npy")):
        embeddings = sim_model.encode(sentences, device=device, batch_size=args.batch_size, normalize_to_unit=True, keepdim=True, return_numpy=True)

        print("============ writting embeddings ============")
        disk_embeddings = np.memmap(os.path.join(args.index, "query.embeddings.npy"),
                                    dtype=np.float32,
                                    mode="w+",
                                    shape=(embeddings.shape[0], embeddings.shape[1]))
        disk_embeddings[:] = embeddings[:]
    else:
        embeddings = np.memmap(os.path.join(args.index, "query.embeddings.npy"),
                               dtype=np.float32,
                               mode="r",
                               shape=(len(sentences), 1024))
    
    print("============ reading index ============")
    index = faiss.read_index(os.path.join(args.index, args.index_name))

    # if device == "cuda":
    #     res = faiss.StandardGpuResources()
    #     res.setTempMemory(20 * 1024 * 1024 * 1024)
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
            
    def pack_single_result(dist, idx):
        return [(int(i), float(s)) for i, s in zip(idx, dist) if s >= args.threshold]

    results = []
    batch_size = args.batch_size
    pbar = tqdm(total=len(sentences))
    start_ = 0
    while start_ < len(sentences):
        end_ = min(start_+batch_size, len(sentences))
        top_value, top_index = index.search(embeddings[start_:end_].astype(np.float32), args.top_k * args.top_k)
        
        for idx_ in range(end_-start_):
            results.append(pack_single_result(top_value[idx_], top_index[idx_]))
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()

    write_file(args.output, results)
               
if __name__ == '__main__':
    main()