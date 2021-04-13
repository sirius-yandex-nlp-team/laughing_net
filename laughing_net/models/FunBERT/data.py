import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, BertTokenizer

from laughing_net.config import params
from laughing_net.context import ctx


class FunDataset:
    def __init__(self, frame, kind="train", model_type="bert"):
        assert kind in ["train", "val", "test"]
        self.frame = frame
        self.kind = kind
        self.model_type = model_type
        self.present_indices = self.frame.index.get_level_values(0).unique()

        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __tokenize(self, sentence: str, org: bool):
        if self.model_type == "bert":
            sentence = f"<s> {sentence} </s>"
        elif self.model_type == "roberta":
            sentence = f"[CLS] {sentence} [CLS]"

        tokenized_text = self.tokenizer.tokenize(sentence)

        sentence = self.tokenizer.convert_tokens_to_ids(sentence)

        if self.model_type == 'bert':
            if org:
                entity_locs = [i for i, s in enumerate(
                    tokenized_text) if '<' in s]
            else:
                entity_locs = [i for i, s in enumerate(
                    tokenized_text) if '^' in s]
        if self.model_type == 'roberta':
            if org:
                entity_locs = [i for i, s in enumerate(
                    tokenized_text) if '<' in s and len(s) == 2]
            else:
                entity_locs = [i for i, s in enumerate(
                    tokenized_text) if '^' in s and len(s) == 2]

        return sentence, entity_locs

    def __getitem__(self, idx: int):
        item = self.frame.loc[self.present_indices[idx]]
        original = item['original'].replace("\"", "")
        edit = item['edit']

        replaced = original[original.index("<"): original.index(">")+1]

        out = dict()

        original_seq = original.replace("<", "< ").replace("/>", " <")
        edited_seq = original.replace(replaced, f"^ {edit} ^")

        out['original_seq'], entity_locs1 = self.__tokenize(original_seq, True)
        out['edited_seq'], entity_locs2 = self.__tokenize(edited_seq, False)

        out['entity_locs'] = np.concatenate((entity_locs1, entity_locs2), 1)

        if self.kind == "train":
            out['target'] = item['meanGrade']

        if self.kind == "val":
            out['target'] = item['meanGrade']
            out['id'] = item['id']

        if self.kind == "test":
            out['id'] = item['id']

        return item

    def __len__(self):
        return len(self.present_indices)

    def __extract_key_from_dict_list(self, d, key):
        return [i[key] for i in d]

    def __pad_and_pack(self, sequence):
        sequence = list(map(torch.tensor, sequence))
        sequence = pad_sequence(
            sequence, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return sequence

    def collate_fn(self, items):
        batch = dict()

        original_seq = self.__extract_key_from_dict_list(items, "original_seq")
        edited_seq = self.__extract_key_from_dict_list(items, "edited_seq")
        batch["original_seq"] = self.__pad_and_pack(original_seq)
        batch["edited_seq"] = self.__pad_and_pack(edited_seq)

        entity_locs = self.__extract_key_from_dict_list(items, "entity_locs")
        batch["entity_locs"] = torch.tensor(entity_locs)

        if "target" in items[0]:
            target = self.__extract_key_from_dict_list(items, "target")
            batch["target"] = torch.tensor(target)

        if "id" in items[0]:
            id = self.__extract_key_from_dict_list(items, "id")
            batch["id"] = torch.tensor(id)

        return batch


class FunDataModule(pl.LightningDataModule):
    def __init__(self, model_type):
        super(FunDataModule, self).__init__()
        self.model_type = model_type

    def train_dataloader(self):
        train_base = pd.read_csv(
            ctx.data_dir / params.data.raw.task_1.train, sep=",", index_col=0)
        train_finelines = pd.read_csv(
            ctx.data_dir / params.data.raw.task_1.funlines, sep=",", index_col=0)

        train_frame = pd.concat(
            [train_base, train_finelines], ignore_index=True)

        self.train_dataset = FunDataset(
            frame=train_frame,
            kind="train",
            model_type=self.model_type
        )

        sampler = RandomSampler(self.train_dataset)

        return DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=params.data.train_batch_size,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        val_frame = pd.read_csv(
            ctx.data_dir / params.data.raw.task_1.val, sep=",", index_col=0)
        self.val_dataset = FunDataset(
            frame=val_frame,
            kind="val",
            model_type=self.model_type
        )

        sampler = SequentialSampler(self.val_dataset)

        return DataLoader(
            dataset=self.val_dataset,
            sampler=sampler,
            batch_size=params.data.train_batch_size,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        test_frame = pd.read_csv(
            ctx.data_dir / params.data.raw.task_1.test, sep=",", index_col=0)
        self.test_dataset = FunDataset(
            frame=test_frame,
            kind="test",
            model_type=self.model_type
        )

        sampler = SequentialSampler(self.test_dataset)

        return DataLoader(
            dataset=self.test_dataset,
            sampler=sampler,
            batch_size=params.data.test_batch_size,
            collate_fn=self.test_dataset.collate_fn
        )
