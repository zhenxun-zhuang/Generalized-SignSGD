First run `getdata.sh` to acquire the Penn Treebank dataset

Example commands:

```
python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 5 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method GeneralizedSignSGD --weight-decay 1.2e-6 --momentum 0.9 --eta0 0.001 --beta2 0.999 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs

python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 750 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method Adam --weight-decay 5e-6 --momentum 0.9 --eta0 0.002 --beta2 0.999 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs

python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 750 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method SGD --eta0 1 --momentum 0.9 --weight-decay 1e-5 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs

python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 750 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method SGDClipGrad --eta0 50 --momentum 0 --weight-decay 1.2e-6 --clipping-param 10 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs

python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 750 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method SGDClipMomentum --eta0 20 --momentum 0.9 --weight-decay 1.2e-6 --clipping-param 2.5 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs

python ./main_lstm.py --dataroot ./data/penn --dataset PennTreebank --model LSTM --epochs 750 --batch-size 40 --dropouti 0.4 --dropouth 0.25 --optim-method SGDNormalized --eta0 2 --momentum 0.9 --weight-decay 5e-6 --gpu-id 0 --reproducible --seed 2022 --log-folder ./logs
```
