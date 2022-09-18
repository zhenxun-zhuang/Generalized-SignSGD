import math
import torch
import torch.nn.functional as F


def get_model_grads(model):
    return [p.grad.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_params(model):
    return [p.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_named_param_grads(model):
    return {name: param.grad.clone().detach() for name, param in model.named_parameters()}

def comp_model_param_diff(model1, model2):
    model1_named_params = {name: param for name, param in model1.named_parameters()}
    model_param_diff_dict = {}
    for model2_param_name, model2_param in model2.named_parameters():
        model_param_diff_dict[model2_param_name] = model1_named_params[model2_param_name] - model2_param
    return model_param_diff_dict


def norm_diff(tensor_list1, tensor_list2=None):
    if tensor_list2 is None:
        tensor_list2 = [0] * len(tensor_list1)
    assert len(tensor_list1) == len(tensor_list2)
    return math.sqrt(sum((tensor_list1[i]-tensor_list2[i]).norm(2)**2 for i in range(len(tensor_list1))))


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold
