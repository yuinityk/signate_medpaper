import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import transformers

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    pass

sys.path.append('./src/')
from data.Dataset import ClassifyDataset, SRTitleDataset, SRTitleAbstConcatenateDataset
from training.Model import SRTitleClassifyTransformer
from utils.ExpConfig import SubmitArgs
from utils.seed_torch import seed_torch


def batch_iterate(args, model, loader):
    out_dict = {}
    with torch.no_grad():
        t = tqdm(test_loader)
        for i, (id_, input_ids_title, attention_mask_title) in enumerate(t):
            input_ids_title = input_ids_title.to(args.device)
            attention_mask_title = attention_mask_title.to(args.device)

            output = model(
                input_ids_title=input_ids_title,
                attention_mask_title=attention_mask_title
            )
            for id_tmp, out_tmp in zip(id_.numpy(), output.detach().cpu().numpy()):
                out_dict[id_tmp] = out_tmp

    return out_dict

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


def test_base(df_test, args, get_dataset_func, base_model, fold=None):
    test_loader = get_dataset_func(df_test, args)
    model = load_model(args, base_model, fold)
    out_dict = batch_iterate(args, model, test_loader)
    return out_dict

def test_title_abst_concat(df_test, args, fold):

    def get_dataset_func_title_abst_concat(df_test, args):

        test_dataset = SRTitleAbstConcatenateDataset(
            df_test,
            args.model_name,
            max_length=args.max_length,
            train=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

        return test_loader

    base_model = SRTitleClassifyTransformer
    out_dict = test_base(df_test, args, get_dataset_func_title_abst_concat, base_model,
                         fold)
    prob_test = pd.DataFrame(out_dict.values(), index=out_dict.keys()).rename({0: f"prob_{fold}"})

    return prob_test

def test_title_abst_concat_nofold(df_test, args):

    def get_dataset_func_title_abst_concat(df_test, args):
        test_dataset = SRTitleAbstConcatenateDataset(
            df_test,
            args.model_name,
            max_length=args.max_length,
            train=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

        return test_loader

    base_model = SRTitleClassifyTransformer
    out_dict = test_base(df_test, args, get_dataset_func_title_abst_concat, base_model)
    prob_test = pd.DataFrame(out_dict.values(), index=out_dict.keys()).rename({0: f"prob"})

    return prob_test


def ensemble_folds(prob_test, args, ensemble_type):
    prob_array = prob_test.values

    if ensemble_type == "prob_mean":
        prob_array = np.mean(prob_array, axis=1)
        pred_array = np.where(prob_array > args.thr, 1, 0)

    elif ensemble_type == "pred_vote":
        pred_array = np.where(prob_array > args.thr, 1, 0)
        pred_array = np.mean(pred_array, axis=1)
        pred_array = np.where(pred_array >= 0.5, 1, 0)

    df_pred = pd.DataFrame(pred_array, index=prob_test.index)

    return df_pred


def main_title_abst_concat(args):
    seed_torch(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    dataset = ClassifyDataset(dir_path=args.dir_path)
    df_train = dataset.get(args.fname_train, args.id_to_1, args.id_to_0)
    df_test = dataset.get(args.fname_test)
    if args.thr is None:
        thr = df_train['judgement'].mean()
        args.thr = thr

    prob_test = None
    for fold in range(args.kfold_nsplit):
        _prob_test_fold = test_title_abst_concat(df_test, args, fold)
        if prob_test is None:
            prob_test = _prob_test_fold.copy()
        else:
            prob_test = prob_test.merge(_prob_test_fold, left_index=True, right_index=True)

    prob_test.to_csv(os.path.join(args.save_path, f"prob_test_{args.trial_name}.csv"), index=True, header=True)

    for ensemble_type in ["prob_mean", "pred_vote"]:
        df_pred = ensemble_folds(prob_test, args, ensemble_type).sort_index()
        df_pred.to_csv(os.path.join(args.save_path, f"submit_{args.trial_name}_{ensemble_type}.csv"), index=True, header=False)


def main_title_abst_concat_nofold(args):
    seed_torch(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    dataset = ClassifyDataset(dir_path=args.dir_path)
    df_train = dataset.get(args.fname_train, args.id_to_1, args.id_to_0)
    df_test = dataset.get(args.fname_test)
    if args.thr is None:
        thr = df_train['judgement'].mean()
        args.thr = thr

    prob_test = test_title_abst_concat_nofold(df_test, args)
    prob_test.to_csv(os.path.join(args.save_path, f"prob_test_{args.trial_name}.csv"), index=True, header=True)

    for ensemble_type in ["prob_mean", "pred_vote"]:
        df_pred = ensemble_folds(prob_test, args, ensemble_type).sort_index()
        df_pred.to_csv(os.path.join(args.save_path, f"submit_{args.trial_name}_{ensemble_type}.csv"), index=True, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname_args')
    fname_args = parser.parse_args().fname_args

    with open(fname_args, 'r') as f:
        args = json.load(f)
    main_funcname = args['main_funcname']
    args = SubmitArgs(**args)
    if args.device == 'tpu':
        args.device = xm.xla_device()
    eval(main_funcname)(args)
