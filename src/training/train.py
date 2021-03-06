import os
import sys
import time
import json
import argparse
from functools import partial
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
import transformers

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    pass

sys.path.append('./src/')
from data.Dataset import (
    ClassifyDataset,
    SRTitleDataset,
    SRTitleAbstConcatenateDataset,
    SRTitleAbstConcatenateSupervisedCLDataset,
)
from training.Model import (
    SRTitleClassifyTransformer,
    SRTitleEmbedTransformer,
)
from training.Loss import MaxMarginContrastiveLoss
from utils.ExpConfig import Args
from utils.Meter import AverageMeter, MetricMeter
from utils.seed_torch import seed_torch


def step_epoch(args, model, loader, criterion, phase, epoch, batch_func, optimizer=None, scheduler=None):
    print(f'{phase} epoch')
    losses = AverageMeter()
    scores = MetricMeter(args.thr)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    if phase == 'train':
        scores_avg, losses_avg = batch_func(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores)
    else:
        with torch.no_grad():
            scores_avg, losses_avg = batch_func(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores)

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


def batch_iterate_SCL(args, model, loader, criterion, phase, epoch, optimizer, scheduler, losses, scores):
    t = tqdm(loader)
    for i, (input_ids_0, input_ids_1, attention_mask_0, attention_mask_1, label_0, label_1) in enumerate(t):
        if phase == 'train':
            optimizer.zero_grad()

        input_ids_0 = input_ids_0.to(args.device)
        input_ids_1 = input_ids_1.to(args.device)
        attention_mask_0 = attention_mask_0.to(args.device)
        attention_mask_1 = attention_mask_1.to(args.device)
        label_0 = label_0.to(args.device)
        label_1 = label_1.to(args.device)

        embed_0 = model(
            input_ids_title=input_ids_0,
            attention_mask_title=attention_mask_0
        )
        embed_1 = model(
            input_ids_title=input_ids_1,
            attention_mask_title=attention_mask_1
        )

        loss = criterion(embed_0, embed_1, label_0, label_1)
        if phase == 'train':
            loss.backward()
            if type(args.device) != str: # TPU
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
            if scheduler and args.step_scheduler:
                scheduler.step()

        batch_size = label_0.size(0)
        losses.update(loss.item(), batch_size)

    t.close()

    return None, losses.avg


def load_model(args, base_model, fold=None):
    config = transformers.AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 1
    if args.dropout is not None:
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout

    model = base_model(args.model_name, config=config)

    # TODO CUDA <-> TPU transferability
    # TODO simplify model load
    if args.base_model_name:
        base_file_path = f'./output/{args.base_model_name}/{args.base_model_name}-fold_{fold}.bin'
        model.load_state_dict(torch.load(base_file_path))

    elif args.base_all_model_path:
        base_file_path = f'./output/{args.base_all_model_path}.bin'
        model.load_state_dict(torch.load(base_file_path))

    if args.model_adhoc:
        model = eval(args.model_adhoc)(model)

    model.to(args.device)

    return model

def model_adhoc_freeze_bert(model):
    for param in model.transformer_title.bert.parameters():
        param.requires_grad = False

    return model


def train_cv_base(df_train, args, fold, get_dataset_func, base_model, batch_iterate_func):
    train_fold = df_train[df_train['fold']!=fold]
    valid_fold = df_train[df_train['fold']==fold]

    if args.debug:
        train_fold = train_fold.sample(frac=0.05, random_state=args.seed)
        valid_fold = valid_fold.sample(frac=0.05, random_state=args.seed)

    train_fold = train_fold.reset_index(drop=True)
    valid_fold = valid_fold.reset_index(drop=True)

    train_loader, valid_loader = get_dataset_func(train_fold, valid_fold, args)

    model = load_model(args, base_model, fold=fold)

    criterion = eval(args.loss)()

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)

    num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    num_training_steps = int(args.epochs * len(train_loader))
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_fbeta = 0.
    for epoch in range(args.start_epoch, args.epochs):

        train_avg, train_loss = step_epoch(
            args, model, train_loader, criterion, 'train', epoch,
            batch_iterate_func, optimizer, scheduler
        )

        valid_avg, valid_loss = step_epoch(
            args, model, valid_loader, criterion, 'valid', epoch,
            batch_iterate_func
        )

        if args.epoch_scheduler:
            scheduler.step()

        content = f'''
            {time.ctime()} \n
            Fold:{fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
            Train Loss:{train_loss:0.4f} - FBeta:{train_avg['FBeta']:0.4f}\n
            Valid Loss:{valid_loss:0.4f} - FBeta:{valid_avg['FBeta']:0.4f}\n
        '''
        print(content)

        if valid_avg['FBeta'] > best_fbeta:
            print(f'########## >>>>>>>> Model Improved From {best_fbeta} ----> {valid_avg["FBeta"]}')
            torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.trial_name}-fold_{fold}.bin'))
            best_fbeta = valid_avg['FBeta']


def train_nocv_base(df_train, args, get_dataset_func, base_model, criterion, batch_iterate_func):

    if args.debug:
        pass

    train_loader = get_dataset_func(df_train, args)

    model = load_model(args, base_model)

    criterion = criterion # TODO include criterion in args

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)

    num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    num_training_steps = int(len(train_loader) * args.epochs)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(args.start_epoch, args.epochs):
        train_avg, train_loss = step_epoch(args, model, train_loader,
                                           criterion, 'train', epoch,
                                           batch_iterate_func,
                                           optimizer, scheduler)

        if args.epoch_scheduler:
            scheduler.step()

        content = f'''
            {time.ctime()} \n
            Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
            Train Loss:{train_loss:0.4f} - FBeta:{train_avg['FBeta']:0.4f}\n
        '''
        print(content)
        torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.trial_name}-epoch_{epoch}.bin'))

def train_valid_fold_title(df_train, args, fold):

    def get_dataset_func_title(train_fold, valid_fold, args):
        train_dataset = SRTitleDataset(
            train_fold,
            args.model_name,
            train=True
        )

        valid_dataset = SRTitleDataset(
            valid_fold,
            args.model_name,
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

        return train_loader, valid_loader

    base_model = SRTitleClassifyTransformer
    batch_iterate_func = batch_iterate_title

    train_cv_base(df_train, args, fold, get_dataset_func_title, base_model,
                          batch_iterate_func)


def train_valid_fold_title_abst_concat(df_train, args, fold):

    def get_dataset_func_title_abst_concat(train_fold, valid_fold, args):

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

        return train_loader, valid_loader

    base_model = SRTitleClassifyTransformer
    batch_iterate_func = batch_iterate_title

    train_cv_base(df_train, args, fold, get_dataset_func_title_abst_concat,
                          base_model, batch_iterate_func)


def train_valid_fold_title_abst_concat_supervised_CL(df_train, df_idx_1_1, df_idx_1_0, df_idx_0_0, args):

    def get_dataset_func_scl(df_train, args, df_idx_1_1, df_idx_1_0, df_idx_0_0):

        train_dataset = SRTitleAbstConcatenateSupervisedCLDataset(
            df_train,
            df_idx_1_1,
            df_idx_1_0,
            df_idx_0_0,
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

        return train_loader

    get_dataset_func_ = partial(get_dataset_func_scl, df_idx_1_1=df_idx_1_1, 
                                df_idx_1_0=df_idx_1_0, df_idx_0_0=df_idx_0_0)

    base_model = SRTitleEmbedTransformer

    criterion = MaxMarginContrastiveLoss(margin=args.margin, device=args.device)

    batch_iterate_func = batch_iterate_SCL

    train_nocv_base(df_train, args, get_dataset_func_, base_model, criterion, batch_iterate_SCL)


def train_valid_title_abst_concat_nofold(df_train, args):

    def get_dataset_func_title_abst_concat(df_train, args):
        df_train_ = df_train.copy().reset_index(drop=True)

        train_dataset = SRTitleAbstConcatenateDataset(
            df_train_,
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

        return train_loader

    base_model = SRTitleClassifyTransformer

    criterion = eval(args.loss)()

    batch_iterate_func = batch_iterate_title

    train_nocv_base(df_train, args, get_dataset_func_title_abst_concat, base_model,
                    criterion, batch_iterate_func)


def main_cv_base(args, train_cv_func):
    seed_torch(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    dataset = ClassifyDataset(dir_path=args.dir_path)
    df_train = dataset.get(args.fname_train, args.id_to_1, args.id_to_0)
    if args.thr is None:
        thr = df_train['judgement'].mean()
        args.thr = thr

    kf = StratifiedKFold(n_splits=args.kfold_nsplit, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(df_train, df_train['judgement'])):
        df_train.loc[val_index, 'fold'] = fold

    for fold in range(args.kfold_nsplit):
        train_cv_func(df_train, args, fold)

def main_nocv_base(args, train_nocv_func):
    seed_torch(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    dataset = ClassifyDataset(dir_path=args.dir_path)
    df_train = dataset.get(args.fname_train, args.id_to_1, args.id_to_0)
    if args.thr is None:
        thr = df_train['judgement'].mean()
        args.thr = thr

    train_nocv_func(df_train, args)

def main_title(args):
    main_cv_base(args, train_valid_fold_title)

def main_title_abst_concat(args):
    main_cv_base(args, train_valid_fold_title_abst_concat)

def main_title_abst_concat_supervised_CL(args):

    df_idx_1_1 = pd.read_csv(args.path_idx_1_1)
    df_idx_1_0 = pd.read_csv(args.path_idx_1_0)
    df_idx_0_0 = pd.read_csv(args.path_idx_0_0)

    main_nocv_base(args, partial(
        train_valid_fold_title_abst_concat_supervised_CL,
        df_idx_1_1=df_idx_1_1,
        df_idx_1_0=df_idx_1_0,
        df_idx_0_0=df_idx_0_0
    ))

def main_title_abst_concat_nofold(args):

    main_nocv_base(args, train_valid_title_abst_concat_nofold)


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



