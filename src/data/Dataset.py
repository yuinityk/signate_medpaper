import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ClassifyDataset(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def get(self, fname, id_to_1=None, id_to_0=None):
        file_path = os.path.join(self.dir_path, fname)
        self.df_ = pd.read_csv(file_path)

        self.modify_judgement(id_to_1, id_to_0)

        return self.df_

    def modify_judgement(self, id_to_1=None, id_to_0=None):
        if id_to_1 is not None:
            cond = self.df_['id'].isin(id_to_1)
            self.df_.loc[cond, 'judgement'] = 1
        if id_to_0 is not None:
            cond = self.df_['id'].isin(id_to_0)
            self.df_.loc[cond, 'judgement'] = 0


class SRTitleDataset(Dataset):
    def __init__(self, df, model_name, train=True):
        self.df = df
        self.model_name = model_name
        self.train = train
        self.labels = None

        if train:
            self.labels = df['judgement'].values
        else:
            pass

        self.title = df['title'].tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.title_tokenized = tokenizer.batch_encode_plus(
            self.title,
            padding='max_length',
            max_length=72,
            truncation=True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df['id'][idx]
        input_ids_title = torch.tensor(self.title_tokenized['input_ids'][idx])
        attention_mask_title = torch.tensor(self.title_tokenized['attention_mask'][idx])

        if self.train:
            label = torch.tensor(self.labels[idx]).float()
            return id_, input_ids_title, attention_mask_title, label
        else:
            return id_, input_ids_title, attention_mask_title


class SRTitleAbstConcatenateDataset(Dataset):
    def __init__(self, df, model_name, max_length=512, train=True):
        self.df = df
        self.model_name = model_name
        self.train = train
        self.labels = None

        if train:
            self.labels = df['judgement'].values
        else:
            pass

        self.text = (df['title']+df['abstract'].fillna('')).tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_tokenized = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df['id'][idx]
        input_ids_title = torch.tensor(self.text_tokenized['input_ids'][idx])
        attention_mask_title = torch.tensor(self.text_tokenized['attention_mask'][idx])

        if self.train:
            label = torch.tensor(self.labels[idx]).float()
            return id_, input_ids_title, attention_mask_title, label
        else:
            return id_, input_ids_title, attention_mask_title
