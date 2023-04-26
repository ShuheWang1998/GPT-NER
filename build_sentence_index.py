import argparse
import os
from tqdm import tqdm
import faiss
from simcse import SimCSE
import numpy as np


def read_file(file_path):
    sentences = []
    print("============ reading ============")
    file = open(file_path, "r")
    for line in file:
        sentences.append(line.strip())
    file.close()
    return sentences

def main():
    parser = argparse.ArgumentParser(description='build indexes for similarity search')
    parser.add_argument("--data", type=str, required=True, help="paths to sentences")
    parser.add_argument("--output", type=str, required=True, help="paths to output index")
    parser.add_argument("--output-name", type=str, required=True, help="file name for the output file")
    parser.add_argument("--batch-size", type=int, default=64, required=False, help="batch size for encoding sentences")
    parser.add_argument("--model-path", type=str, required=True, help="paths for simcse model")
    parser.add_argument('--use-gpu', default=False, action='store_true', help='if true, use gpu for building')
    parser.add_argument('--faiss-fast', default=False, action='store_true',
                                        help='whether to use the fast mode of faiss, and it might cause precision lost')
    args = parser.parse_args()

    sentences = read_file(args.data)

    sim_model = SimCSE(args.model_path)
    
    print("============ computing embeddings ============")
    device = "cuda" if args.use_gpu else "cpu"
    
    if not os.path.exists(os.path.join(args.output, "datastore.embeddings.npy")):
        embeddings = sim_model.encode(sentences, device=device, batch_size=args.batch_size, normalize_to_unit=True, return_numpy=True)

        print("============ writting embeddings ============")
        disk_embeddings = np.memmap(os.path.join(args.output, "datastore.embeddings.npy"),
                                    dtype=np.float32,
                                    mode="w+",
                                    shape=(embeddings.shape[0], embeddings.shape[1]))
        disk_embeddings[:] = embeddings[:]
    else:
        embeddings = np.memmap(os.path.join(args.output, "datastore.embeddings.npy"),
                               dtype=np.float32,
                               mode="r",
                               shape=(len(sentences), 1024))

    print("============ building index ============")
    
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    if args.faiss_fast:
        # 100 is a default setting in simcse
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(100, len(sentences)))
    else:
        index = quantizer

    # if device == "cuda":
    #     print("using gpu for faiss ...")
    #     res = faiss.StandardGpuResources()
    #     res.setTempMemory(20 * 1024 * 1024 * 1024)
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    
    if args.faiss_fast:
        index.train(embeddings.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    # 10 is a default setting in simcse
    index.nprobe = min(10, len(sentences))
    
    if device == "cuda":
        index = faiss.index_gpu_to_cpu(index)

    faiss.write_index(index, os.path.join(args.output, f"index.{args.output_name}"))

    # index = faiss.read_index(os.path.join(args.output, f"index.{args.output_name}"))
    # res = faiss.StandardGpuResources()
    # # res.setTempMemory(20 * 1024 * 1024 * 1024)
    # newindex = faiss.index_cpu_to_gpu(res, 0, index)

    # # query_embedding = sim_model.encode(["Food: Where European inflation slipped up"], device=device, batch_size=args.batch_size, normalize_to_unit=True, return_numpy=True)

    # sentences = []
    # file = open("/nfs1/shuhe/gpt3-nmt/data/en-fr/dev.en", "r")
    # for line in file:
    #     sentences.append(line.strip())
    # file.close()

    # query_embedding = sim_model.encode(sentences, device=device, batch_size=args.batch_size, normalize_to_unit=True, return_numpy=True)

    # distance, idx = newindex.search(query_embedding[:20].astype(np.float32), 5)
    # print(distance, idx)

if __name__ == '__main__':
    main()