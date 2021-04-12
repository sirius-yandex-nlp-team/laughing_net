import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torchnlp.nn as nn_nlp
import pytorch_lightning as pl
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, RobertaModel, BertModel, DistilBertModel, RobertaForMaskedLM

from laughing_net.models.FunBERT.data import get_bert_lm_dataloader, get_dataloaders_bert, get_sent_emb_dataloaders_bert
from laughing_net.logger import logger

class LightBERT(pl.LightningModule):
    def __init__(self, lr: float, model_type: str, lm_pretrain: str):
        super(LightBERT, self).__init__()

        if lm_pretrain and model_type == 'roberta':
            self.model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'roberta':
            self.model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'bert':
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_type == 'distilbert':
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.lr = lr

        self.attention = nn_nlp.Attention(768 * 2)
        self.prelu = nn.PReLU()
        self.linear_reg1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 * 8, 1024))
        self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))

        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        self.optimizer = Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.001
            )
        return self.optimizer

    def forward(self, sentence1, sentence2, ent_locs):
        ''' 
        FunBERT forward function

        Parameters
        ----------

        sentence1: torch.Tensor - the first sentence
        sentence2: torch.Tensor - the second sentence
        ent_locs: torch.Tensor - the entity locations
        '''

        
        attn_mask0 = (sentence1 != self.tokenizer.pad_token_id).long()
        output_per_seq1, _, attention_layer_inps = self.model(sentence1.long(), attention_mask=attn_mask0)
        output_per_seq1 = torch.cat((output_per_seq1, attention_layer_inps[11]), 2)
        attn_mask1 = (sentence2 != self.tokenizer.pad_token_id).long()
        output_per_seq2, _, attention_layer_inps = self.model(sentence2.long(), attention_mask=attn_mask1)
        output_per_seq2 = torch.cat((output_per_seq2, attention_layer_inps[11]))

        final_scores = []
        for (i, loc) in enumerate(ent_locs):
            # +1 is to ensure that the symbol token is not considered

            entity1 = torch.mean(output_per_seq1[i, loc[0] + 1:loc[1]], 0)
            entity2 = torch.mean(output_per_seq2[i, loc[2] + 1:loc[3]], 0)

            imp_seq1 = torch.cat((output_per_seq1[i, 0:loc[0] + 1], output_per_seq1[i, loc[1]:]), 0)
            imp_seq2 = torch.cat((output_per_seq2[i, 0:loc[2] + 1], output_per_seq2[i, loc[3]:]), 0)
            _, attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0), imp_seq2.unsqueeze(0))
            sent_attn = torch.sum(attention_score.squeeze(0).expand(768 * 2, -1).t() * imp_seq2, 0)
            _, attention_score1 = self.attention(entity1.unsqueeze(0).unsqueeze(0), imp_seq1.unsqueeze(0))
            sent_attn1 = torch.sum(attention_score1.squeeze(0).expand(768 * 2, -1).t() * imp_seq1, 0)
            sent_out = self.prelu(self.linear_reg1(torch.cat((sent_attn, sent_attn1, output_per_seq2[i, 0], entity2), 0)))
            final_out = self.final_linear(sent_out)
            final_scores.append(final_out)

        return torch.stack((final_scores))

    def training_step(self, batch):
        ''' 
        Step of the training loop

        Parameters
        ----------

        batch: dict
        + sentence1 - the first sentence
        + sentence2 - the second sentence
        + ent_locs - the entity locations
        + target - ground truth
        '''

        target = batch['target']

        scores = self.forward(batch)
        loss = self.criterion(scores, target)

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        logger.log_metric("train_loss", loss.item())
        return {"loss": loss}

    def train_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        for key, value in logs.items():
            logger.log_metric(key, value.item(), dvc=True)

        return

    def validation_step(self, batch):
        ''' 
        Step of the validation loop

        Parameters
        ----------

        batch: dict
        + sentence1 - the first sentence
        + sentence2 - the second sentence
        + ent_locs - the entity locations
        + target - ground truth
        '''

        target = batch['target']

        scores = self.forward(batch)
        loss = self.criterion(scores, target)

        # TODO Add model save

        return {"val_loss": loss}
    
    def validation_end(self, outputs):
        valid_loss_mean = torch.stack([x["valid_loss"] for x in outputs]).mean()
        logs = {
            "valid_loss": valid_loss_mean,
        }

        for key, value in logs.items():
            logger.log_metric(key, value.item(), dvc=True)

        return

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader

    def criterion(self, prediction, target):
        return self.loss(prediction.squeeze(1), target)
