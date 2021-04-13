from typing import Dict, List
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import torchnlp.nn as nn_nlp
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import mean_squared_error
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, RobertaModel, BertModel, DistilBertModel

from laughing_net.logger import logger


class FunBERT(pl.LightningModule):
    r'''
        Constructor of FunBERT model.

        Args:
            lr (`float`):
                Learning rate for optimizer
            model_type (`str` `'bert'` | `'roberta'`. Default value is `'roberta'`)
                Type of used pretrained transformer 
    '''

    def __init__(self, lr: float, model_type: str = "bert"):
        super(FunBERT, self).__init__()

        if model_type == 'roberta':
            self.model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'bert':
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.lr = lr

        self.attention = nn_nlp.Attention(768 * 2)
        self.prelu = nn.PReLU()
        self.linear_reg1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 * 8, 1024))
        self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))

        self.loss = mean_squared_error

    def configure_optimizers(self) -> Optimizer:
        '''
        Optimizer initialization function

        Returns:
        `torch.optim.Optimizer`
            Optimizer for model
        '''

        self.optimizer = Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.001
        )
        return self.optimizer

    def forward(self, original_seq: torch.Tensor, edited_seq: torch.Tensor, ent_locs: torch.Tensor) -> torch.Tensor:
        ''' 
        FunBERT forward function

        Args:
            original_seq (`torch.Tensor`)
                The sentance with original word
            edited_seq (`torch.Tensor`)
                The sentance with edited word (joke)
            ent_locs (`torch.Tensor`)
                The entity locations

        Returns:
            `torch.Tensor`
                Humor score vector 

        '''

        attn_mask0 = (original_seq != self.tokenizer.pad_token_id).long()
        output_per_seq1, _, attention_layer_inps = self.model(original_seq.long(), attention_mask=attn_mask0).values()
        output_per_seq1 = torch.cat((output_per_seq1, attention_layer_inps[11]), 2)

        attn_mask1 = (edited_seq != self.tokenizer.pad_token_id).long()
        output_per_seq2, _, attention_layer_inps = self.model(edited_seq.long(), attention_mask=attn_mask1).values()
        output_per_seq2 = torch.cat((output_per_seq2, attention_layer_inps[11]), 2)

        final_scores = []
        for (i, loc) in enumerate(ent_locs):

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

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        ''' 
        Step of the training loop

        Args:
            batch (`Dict`)
            + `original_seq` - The sentance with original word
            + `edited_seq` - The sentance with edited word (joke)
            + `entity_locs` - The entity locations
            + `target` - Vector of truth humor score

            batch_idx (`int`)
                Index of a batch

        Returns:
            `Dict`
                Train loss
        '''

        original_seq = batch['original_seq']
        edited_seq = batch['edited_seq']
        entity_locs = batch['entity_locs']
        target = batch['target']

        scores = self.forward(original_seq, edited_seq, entity_locs)
        # loss = self.criterion(scores, target)
        loss = self.criterion(scores, target)

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        logger.log_metric("train_loss", loss.item())
        return {"loss": loss}

    def train_end(self, outputs: List[Dict]) -> None:
        '''
        End of the training loop

        Args:
            outputs (`List[Dict]`) 
                List of losses
        '''

        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        for key, value in logs.items():
            logger.log_metric(key, value.item(), dvc=False)

        return

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        ''' 
        Step of the validation loop

        Args:
            batch (`dict`)
            + `original_seq` - The sentance with original word
            + `edited_seq` - The sentance with edited word (joke)
            + `entity_locs` - The entity locations
            + `target` - Vector of truth humor score

            batch_idx (`int`)
                Index of a batch

        Returns:
            `Dict`
                Validation loss
        '''

        original_seq = batch['original_seq']
        edited_seq = batch['edited_seq']
        entity_locs = batch['entity_locs']
        target = batch['target']

        scores = self.forward(original_seq, edited_seq, entity_locs)
        loss = self.criterion(scores, target)

        # TODO Add model save

        return {"val_loss": loss}

    def validation_end(self, outputs: List[Dict]) -> None:
        '''
        End of the validation loop

        Args:
            outputs (`List[Dict]`) 
                List of losses
        '''

        valid_loss_mean = torch.stack(
            [x["valid_loss"] for x in outputs]).mean()
        logs = {
            "valid_loss": valid_loss_mean,
        }

        for key, value in logs.items():
            logger.log_metric(key, value.item(), dvc=False)

        return

    def criterion(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Calculation of the loss function

        Args:
            prediction (`torch.Tensor`)
                Return of the model
            target (`torch.Tensor`)
                Target value

        Returns:
        `torch.Tensor`
            Calculated loss
        '''
        return self.loss(prediction.squeeze(1).float(), target.float())
