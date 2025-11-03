# train bpe tokenizer
python train.py tokenizer -n ts_tokenizer -s 10000 -i /data/nlp_course/pre_TinyStory_Train_Tokenizer.txt
uv run python train.py tokenizer -n owt_tokenizer -s 32000 -i /data/nlp_course/pre_Owt_Train_Tokenizer.txt
# prepare dataset(tokenized data)
time python train.py dataset -t "owt_tokenizer-32000" -i /data/nlp_course/pre_Owt_Train_Tokenizer.txt  > verify.txt
time python train.py dataset -t "ts_tokenizer-10000" -i /data/nlp_course/pre_TinyStory_Train_Tokenizer.txt  > verify.txt
# train
time python train.py model \
--dataset "assets/dataset/ts_tokenizer-10000-pre_TinyStory_Train_Tokenizer.dat" \
--checkpoint "assets/checkpoint" \
--vocab 10000 \
--context 256 \
--dmodel 512 \
--dff 1344 \
--rope 10000 \
--layers 4 \
--heads 16 \
--batch 128 \
--total 3276800