## Image classification experiments on CIFAR-10

Example commands:

```
python ./main.py --optim-method GeneralizedSignSGD --eta0 0.0002 --momentum 0.9 --beta2 0.999 --epsilon 1e-8 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10

python ./main.py --optim-method Adam --eta0 0.0009 --momentum 0.9 --beta2 0.999 --epsilon 1e-8 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10

python ./main.py --optim-method SGD --eta0 0.07 --momentum 0.9 --nesterov --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10

python ./main.py --optim-method SGDClipGrad --eta0 0.5 --momentum 0 --weight-decay 0.0001 --clipping-param 1 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10

python ./main.py --optim-method SGDClipMomentum --eta0 10 --momentum 0.9 --nesterov --weight-decay 0.0001 --clipping-param 0.1 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10

python ./main.py --optim-method SGDNormalized --eta0 0.1 --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs --dataroot ./data --dataset CIFAR10
```
