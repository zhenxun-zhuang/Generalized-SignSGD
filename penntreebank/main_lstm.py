import argparse
import json
import math
import numpy as np
import os
import random
import time
import torch
import torch.optim as optim
from model.splitcross import SplitCrossEntropyLoss
from model.awdlstm import RNNModel
from generalized_signsgd import GeneralizedSignSGD
from sgd_clip import SGDClipGrad
from sgd_normalized import SGDNormalized


###############################################################################
# Make batch
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data_batched = data.view(batch_size, -1).t().contiguous().cuda()
    return data_batched

def get_batch(source, i, bptt, seq_len=None):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

###############################################################################
# Training code
###############################################################################

def evaluate(model, criterion, data_source, bptt, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    avg_loss = total_loss.item() / len(data_source)
    return avg_loss

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = torch.tensor(0.0)
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, model, criterion, optimizer, train_data, val_data, test_data, eval_batch_size, test_batch_size):
    results = {
        'train_losses': [],
        'train_ppl': [],
        'val_losses': [],
        'val_ppl': [],
        'test_losses': [],
        'test_ppl': [],
        'epoch_elasped_times': []
    }

    begin_avg = False
    ax = {}
    avg_cnt = 0
    best_val_loss = []

    t_total = 0

    for epoch in range(0, args.epochs):
        train_data = train_data
        # Turn on training mode which enables dropout.
        hidden = model.init_hidden(args.batch_size)
        epoch_start_time = time.time()
        running_loss_cur_epoch = 0
        total_ite_cur_epoch = 0
        model.train()
        i = 0
        while i < train_data.size(0) - 1 - 1:
            # print(f'Epoch {epoch} Iteration {total_ite_cur_epoch} / {(train_data.size(0) - 1 - 1) / args.bptt}')

            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            if 'clipping_param' in optimizer.param_groups[0]:
                clipping_param2 = optimizer.param_groups[0]['clipping_param']
                optimizer.param_groups[0]['clipping_param'] = clipping_param2 * seq_len / args.bptt
            model.train()
            data, targets = get_batch(train_data, i, args.bptt, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            # hidden = nn.Parameter(hidden)

            optimizer.zero_grad()
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

            loss = raw_loss
            # Activation Regularization
            if args.alpha: loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            optimizer.step()

            running_loss_cur_epoch += raw_loss.data.item()
            optimizer.param_groups[0]['lr'] = lr2
            if 'clipping_param' in optimizer.param_groups[0]:
                optimizer.param_groups[0]['clipping_param'] = clipping_param2

            total_ite_cur_epoch += 1
            i += seq_len
            t_total += 1

        # Evaluation
        epoch_elapsed_time = time.time() - epoch_start_time
        results['train_losses'].append(running_loss_cur_epoch / total_ite_cur_epoch)
        results['train_ppl'].append(math.exp(running_loss_cur_epoch / total_ite_cur_epoch))
        results['epoch_elasped_times'].append(epoch_elapsed_time)

        if begin_avg:
            avg_cnt += 1
            for prm in model.parameters():
                if avg_cnt == 1:
                    ax[prm] = prm.data.clone()
                else:
                    ax[prm].add_(prm.data.sub(ax[prm]).mul(1 / avg_cnt))
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if len(ax) > 0:
                    prm.data.copy_(ax[prm])

            val_loss2 = evaluate(model, criterion, val_data, args.bptt, eval_batch_size)
            val_ppl2 = math.exp(val_loss2)
            results['val_losses'].append(val_loss2)
            results['val_ppl'].append(val_ppl2)
            test_loss2 = evaluate(model, criterion, test_data, args.bptt, test_batch_size)
            test_ppl2 = math.exp(test_loss2)
            results['test_losses'].append(test_loss2)
            results['test_ppl'].append(test_ppl2)
            print(f'| epoch {epoch:3d} | '
                    f'time: {epoch_elapsed_time:5.2f}s | '
                    f'valid loss {val_loss2:7.4f} | valid ppl {val_ppl2:9.3f} | valid bpc {val_loss2 / math.log(2):8.3f} | '
                    f'test loss {test_loss2:7.4f} | test ppl {test_ppl2:9.3f} | test bpc {test_loss2 / math.log(2):8.3f} |')

            for prm in model.parameters():
                prm.data.copy_(tmp[prm])

        else:
            val_loss = evaluate(model, criterion, val_data, args.bptt, eval_batch_size)
            val_ppl = math.exp(val_loss)
            results['val_losses'].append(val_loss)
            results['val_ppl'].append(val_ppl)
            test_loss = evaluate(model, criterion, test_data, args.bptt, test_batch_size)
            test_ppl = math.exp(test_loss)
            results['test_losses'].append(test_loss)
            results['test_ppl'].append(test_ppl)
            print(f'| epoch {epoch:3d} | '
                    f'time: {epoch_elapsed_time:5.2f}s | '
                    f'valid loss {val_loss:7.4f} | valid ppl {val_ppl:9.3f} | valid bpc {val_loss / math.log(2):8.3f} | '
                    f'test loss {test_loss:7.4f} | test ppl {test_ppl:9.3f} | test bpc {test_loss / math.log(2):8.3f} |')

            if not begin_avg and len(best_val_loss) > args.nonmono \
                and val_loss > min(best_val_loss[:-args.nonmono]):
                print('Starting averaging')
                begin_avg = True

            best_val_loss.append(val_loss)

    return results

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NLP Distributed Training')
    parser.add_argument('--dataroot', type=str, default='.',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='PennTreebank',
                        help='Which dataset to run on (default: PennTreebank).')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--weight-decay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optim-method', type=str, default='GeneralizedSignSGD',
                        help='optimizer to use (default: GeneralizedSignSGD)')
    parser.add_argument('--eta0', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='momentum')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 in Adam (default: 0.999).')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).')
    parser.add_argument('--epsilon', type=float, default=1e-08,
                        help='used in the denominator of AdamClip')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit of each node (default: 200).')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='batch size of the local node')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')

    parser.add_argument('--clipping-param', type=float, default=1.0,
                        help='Weight decay used in optimizer (default: 1.0).')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU is used in this node (default: 0).')

    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs_single_gpu',
                        help='Where to store results.')

    return parser.parse_args()

def main():
    args = arg_parser()
    args.tied = True

    torch.cuda.set_device(args.gpu_id)
    print(f"| Requested GPU {args.gpu_id} "
          f'| Assigned GPU {torch.cuda.current_device()} |')

    # Set the random seed manually for reproducibility.
    if args.reproducible:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    # Load data
    fn = f'corpus.{args.dataset}.data'
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        from dataset.nlp_data import Corpus
        print('Producing dataset...')
        corpus = Corpus(args.dataroot)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 2
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, test_batch_size)

    # Build the model
    ntokens = len(corpus.dictionary)
    model = RNNModel(args.model,
                     ntokens,
                     args.emsize,
                     args.nhid,
                     args.nlayers,
                     args.dropout,
                     args.dropouth,
                     args.dropouti,
                     args.dropoute,
                     args.wdrop,
                     args.tied,)
    init_model_path = f"{args.dataset}_{args.model}_init_model.pt"
    if os.path.isfile(init_model_path):
        model.load_state_dict(torch.load(init_model_path))
    else:
        torch.save(model.state_dict(), init_model_path)
    model.cuda()

    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    init_criterion_path = f"{args.dataset}_{args.model}_init_criterion.pt"
    if os.path.isfile(init_criterion_path):
        criterion.load_state_dict(torch.load(init_criterion_path))
    else:
        torch.save(criterion.state_dict(), init_criterion_path)
    criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.numel() for x in params)

    # Train
    if args.optim_method == 'SGDClipGrad':
        optimizer = SGDClipGrad(params, lr=args.eta0, momentum=0,
                                weight_decay=args.weight_decay, nesterov=args.nesterov,
                                clipping_param=args.clipping_param)
    elif args.optim_method == 'SGDClipMomentum':
        optimizer = SGDClipGrad(params, lr=args.eta0, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov,
                                clipping_param=args.clipping_param)
    elif args.optim_method == 'SGD':
        optimizer = optim.SGD(params, lr=args.eta0, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optim_method == 'SGDNormalized':
        optimizer = SGDNormalized(params, lr=args.eta0, momentum=args.momentum,
                                  weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optim_method == 'Adam':
        optimizer = optim.Adam(params, lr=args.eta0, betas=(args.momentum, args.beta2), eps=args.epsilon,
                               weight_decay=args.weight_decay, amsgrad=False)
    elif args.optim_method == 'GeneralizedSignSGD':
        optimizer = GeneralizedSignSGD(params, lr=args.eta0, eps=args.epsilon,
                                       betas=(args.momentum, args.beta2),
                                       weight_decay=args.weight_decay,
                                       use_bias_correction=True)


    train_results = train(args, model, criterion, optimizer, train_data, val_data, test_data, eval_batch_size, test_batch_size)

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    log_name = (f'{args.dataset}_{args.model}_{args.optim_method}_'
                + ('Eta0_%g_' % (args.eta0))
                + ('Momentum_%g_' % (args.momentum))
                + (('beta2_%g_' % (args.beta2)) if args.optim_method in ['Adam', 'GeneralizedSignSGD'] else '')
                + (('Eps_%g_' % (args.epsilon)) if args.optim_method in ['Adam', 'GeneralizedSignSGD'] else '')
                + ('WD_%g_' % (args.weight_decay))
                + (('Clipping_%g_' % (args.clipping_param)) if args.optim_method in ['SGDClipGrad', 'SGDClipMomentum'] else '')
                + ('Epoch_%d_Batchsize_%d_' % (args.epochs, args.batch_size))
                + (f'GPU_{args.gpu_id}_Seed_{args.seed}'))
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    print('Finished.')

if __name__ == '__main__':
    main()
