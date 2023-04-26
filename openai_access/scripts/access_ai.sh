SOURCEDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/"
SOURCENAME="test.100"
DATANAME="CONLL"
EXAMPLEDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/"
EXAMPLENAME="test.100.simcse.32"
EXAMPLENUM=8
WRITEDIR="/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/100-results"
WRITENAME="tmp.test"
TRAINNAME="train.dev"

python ./get_results_mrc_knn.py \
    --source-dir $SOURCEDIR --source-name $SOURCENAME \
    --data-name $DATANAME --example-dir $EXAMPLEDIR \
    --example-name $EXAMPLENAME \
    --example-num $EXAMPLENUM --train-name $TRAINNAME \
    --write-dir $WRITEDIR --write-name $WRITENAME
    # --last-results "/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en/results/openai.8.train.sequence.fullprompt"