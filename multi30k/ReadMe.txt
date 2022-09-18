The code is based on https://github.com/jadore801120/attention-is-all-you-need-pytorch.

On the WMT'16 Multimodal Machine Translation task (German-English) (http://www.statmt.org/wmt16/multimodal-task.html).

First, download the spacy language model.
```bash
conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

Next, preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

Example commands for training the model:
```bash
python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method GeneralizedSignSGD --lr-mul 10 --momentum 0.9 --beta2 0.98 --epsilon 1e-9 --weight-decay 0.1 --test

python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method Adam --lr-mul 10 --momentum 0.9 --beta2 0.98 --epsilon 1e-9 --weight-decay 0.1 --test

python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method SGD --lr-mul 1 --momentum 0.9 --weight-decay 1 --test

python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method SGDClipGrad --lr-mul 10 --momentum 0 --clipping-param 1 --weight-decay 1 --test

python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method SGDClipMomentum --lr-mul 1 --momentum 0.9 --clipping-param 1 --weight-decay 1 --test

python ./train.py --data-pkl m30k_deen_shr.pkl --embs-share-weight --proj-share-weight --label-smoothing --output-dir ./logs --batch-size 256 --n-warmup-steps 128000 --epoch 400 --reproducible --seed 2022 --optim-method SGDNormalized --lr-mul 10000 --momentum 0.9 --weight-decay 1 --test
```

Finally, to test a model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model model.chkpt -output prediction.txt
```