MRCDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/"
# MRCNAME="test"
MRCNAME="test.100"
# GPTDIR="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/text-3/"
# GPTNAME="openai.17.knn.train.dev.sequence.fullprompt"
# KNNFILE="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/test.100.verify.knn.jsonl"
GPTDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/100-results/"
GPTNAME="openai.32.knn.sequence.fullprompt"
# KNNFILE="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/test.verify.knn.jsonl"
DATANAME="CONLL"
# WRITEDIR="/nfs1/shuhe/gpt3-ner/gpt3-data/en_conll_2003/text-3"
# WRITENAME="openai.17.knn.train.dev.sequence.fullprompt.knn.18.verified"
WRITEDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/100-results"
WRITENAME="openai.32.knn.sequence.fullprompt.verified"

# for knn in 0 2 4 6 8 10 12 14 16 18 20 22;do

# echo $knn
# WRITENAME="openai.17.knn.train.dev.sequence.fullprompt.knn.${knn}.verified"
# echo $WRITENAME

python ./verify_results.py \
    --mrc-dir $MRCDIR --mrc-name $MRCNAME \
    --gpt-dir $GPTDIR --gpt-name $GPTNAME \
    --data-name $DATANAME \
    --write-dir $WRITEDIR --write-name $WRITENAME
    # --knn-file $KNNFILE --knn-num 18
# done