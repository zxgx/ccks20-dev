#!bin/bash

cd helper
python LoadCorpus.py
python GetSegmentDict.py
python GetChar2Prop.py
python GetMentionDict.py

cd ../src
CUDA_VISIBLE_DEVICES=0,1 python train_ner.py
CUDA_VISIBLE_DEVICES=0,1 python mention_extractor.py
python property_extractor.py
python entity_extractor.py
python entity_filter.py
CUDA_VISIBLE_DEVICES=0,1 python similarity.py
CUDA_VISIBLE_DEVICES=0,1 python tuple_extractor.py
python tuple_filter.py

