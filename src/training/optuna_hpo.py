import os
import sys
import time
import json
import argparse
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
import transformers
import optuna

from torch.nn import BCELoss
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    pass

sys.path.append('./src/')
from data.Dataset import ClassifyDataset, SRTitleDataset, SRTitleAbstConcatenateDataset
from training.Model import SRTitleClassifyTransformer
from utils.ExpConfig import Args
from utils.Meter import AverageMeter, MetricMeter
from utils.seed_torch import seed_torch

def step_epoch(args, model, loader, criterion, phase, epoch, optimizer=None, scheduler=None):
    print(f'{phase} epoch')
    losses = AverageMeter()
    scores = MetricMeter(args.thr)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    if phase == 'train':
        # TODO batch_iterateを可変に
        scores_avg, losses_avg = batch_iterate_title(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores)
    else:
        with torch.no_grad():
            scores_avg, losses_avg = batch_iterate_title(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores)

    return scores_avg, losses_avg


def batch_iterate_title(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores):
    t = tqdm(loader)
    for i, (_, input_ids_title, attention_mask_title, labels) in enumerate(t):
        if phase == 'train':
            optimizer.zero_grad()

        input_ids_title = input_ids_title.to(args.device)
        attention_mask_title = attention_mask_title.to(args.device)
        labels = labels.to(args.device)

        output = model(
            input_ids_title=input_ids_title,
            attention_mask_title=attention_mask_title
        )

        loss = criterion(output, labels)
        if phase == 'train':
            loss.backward()
            if type(args.device) != str: # TPU
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
            if scheduler and args.step_scheduler:
                scheduler.step()

        batch_size = labels.size(0)
        scores.update(labels, output)
        losses.update(loss.item(), batch_size)

    t.close()

    return scores.avg, losses.avg



def train_valid_fold_title_abst_concat(trial, df_train, args, fold):
    train_fold = df_train[df_train['fold']!=fold]
    valid_fold = df_train[df_train['fold']==fold]

    if args.debug:
        train_fold = train_fold.sample(frac=0.05, random_state=args.seed)
        valid_fold = valid_fold.sample(frac=0.05, random_state=args.seed)

    train_fold = train_fold.reset_index(drop=True)
    valid_fold = valid_fold.reset_index(drop=True)

    train_dataset = SRTitleAbstConcatenateDataset(
        train_fold,
        args.model_name,
        max_length=args.max_length,
        train=True
    )

    valid_dataset = SRTitleAbstConcatenateDataset(
        valid_fold,
        args.model_name,
        max_length=args.max_length,
        train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    if type(args.dropout) == dict:
        suggest_method = args.dropout["suggest_method"]
        low = args.dropout["low"]
        high = args.dropout["high"]
        # exec(f"dropout = trial.{suggest_method}('dropout', low={low}, high={high})")
        dropout = trial.suggest_uniform("dropout", low=low, high=high)
    elif type(args.dropout) == float:
        dropout = args.dropout

    config = transformers.AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 1
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

    model = SRTitleClassifyTransformer(args.model_name, config=config)

    if args.base_model_name:
        base_file_path = f"./output/{args.base_model_name}/{args.base_model_name}-fold_{fold}.bin"
        model.load_state_dict(torch.load(base_file_path))

    model.to(args.device)
    criterion = eval(args.loss)()

    lr = None
    if type(args.lr) == dict:
        suggest_method = args.lr["suggest_method"]
        low = args.lr["low"]
        high = args.lr["high"]
        # exec(f"lr = trial.{suggest_method}('lr', low={low}, high={high})")
        lr = trial.suggest_uniform("lr", low=low, high=high)
    else:
        lr = args.lr

    optimizer = transformers.AdamW(model.parameters(), lr=lr)

    num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    num_training_steps = int(len(train_loader) * args.epochs)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_fbeta = 0.
    for epoch in range(args.start_epoch, args.epochs):
        train_avg, train_loss = step_epoch(args, model, train_loader,
                                           criterion, 'train', epoch,
                                           optimizer, scheduler)

        valid_avg, valid_loss = step_epoch(args, model, valid_loader,
                                           criterion, 'valid', epoch)

        if args.epoch_scheduler:
            scheduler.step()

        if valid_avg['FBeta'] > best_fbeta:
            best_fbeta = valid_avg['FBeta']

    return best_fbeta


def objective(trial, df_train, args):

    sum_fbeta = 0.
    for fold in range(args.kfold_nsplit):
        sum_fbeta += train_valid_fold_title_abst_concat(trial, df_train, args, fold)
    sum_fbeta /= args.kfold_nsplit

    return sum_fbeta


def main_title_abst_concat(args):
    seed_torch(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    dataset = ClassifyDataset(dir_path=args.dir_path)
    df_train = dataset.get(args.fname_train, args.id_to_1, args.id_to_0)
    thr = df_train['judgement'].mean()
    args.thr = thr
    df_test = dataset.get(args.fname_test)

    kf = StratifiedKFold(n_splits=args.kfold_nsplit, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(df_train, df_train['judgement'])):
        df_train.loc[val_index, 'fold'] = fold

    study = optuna.create_study(
        study_name="study",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        direction="maximize",
    )

    study.optimize(partial(objective, df_train=df_train, args=args), timeout=2*3600)

    print(f"{study.best_value:.5f}")
    print(study.best_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname_args')
    fname_args = parser.parse_args().fname_args

    with open(fname_args, 'r') as f:
        args = json.load(f)
    main_funcname = args['main_funcname']
    args = Args(**args)
    if args.device == 'tpu':
        args.device = xm.xla_device()
    eval(main_funcname)(args)





