import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchnlp.nn as nn_nlp
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, RobertaModel, BertModel, DistilBertModel, RobertaForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import mean_squared_error

from laughing_net.models.FunBERT.data import get_bert_lm_dataloader, get_dataloaders_bert, get_sent_emb_dataloaders_bert


class FunBERT(nn.Module):

    def __init__(self, train_file_path: str, dev_file_path: str, test_file_path: str, lm_file_path: str,
                 train_batch_size: int,
                 test_batch_size: int, lr: float, lm_weights_file_path: str, epochs: int, lm_pretrain: str,
                 model_path: str, model_type: str):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(FunBERT, self).__init__()
        if lm_pretrain and model_type == 'roberta':
            self.bert_model = RobertaForMaskedLM.from_pretrained(
                'roberta-base', output_hidden_states=True)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'roberta':
            self.bert_model = RobertaModel.from_pretrained(
                'roberta-base', output_hidden_states=True)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'bert':
            self.bert_model = BertModel.from_pretrained(
                'bert-base-uncased', output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_type == 'distilbert':
            self.bert_model = DistilBertModel.from_pretrained(
                'distilbert-base-uncased', output_hidden_states=True)
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')
        self.model_type = model_type
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.lm_file_path = lm_file_path
        self.attention = nn_nlp.Attention(768 * 2)
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.lr = lr

        self.prelu = nn.PReLU()
        self.epochs = epochs
        self.linear_reg1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 * 8, 1024))

        self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))

    def pre_train_bert(self):
        optimizer = optim.Adam(self.model.parameters(), 2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, 62, 620)
        step = 0
        train_dataloader = get_bert_lm_dataloader(self.lm_file_path, 32)
        print("Training LM")
        if torch.cuda.is_available():
            self.bert_model.cuda()
        for epoch in range(2):
            print("Epoch : " + str(epoch))
            for ind, batch in enumerate(train_dataloader):
                step += 1

                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                else:
                    inp = batch[0]

                labels = inp.clone()
                # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
                probability_matrix = torch.full(labels.shape, 0.15)
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                    labels.tolist()
                ]
                probability_matrix.masked_fill_(torch.tensor(
                    special_tokens_mask, dtype=torch.bool), value=0.0)
                if self.tokenizer._pad_token is not None:
                    padding_mask = labels.eq(self.tokenizer.pad_token_id)
                    padding_mask = padding_mask.detach().cpu()
                    probability_matrix.masked_fill_(padding_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                # We only compute loss on masked tokens
                labels[~masked_indices] = -100

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(torch.full(
                    labels.shape, 0.8)).bool() & masked_indices
                inp[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.mask_token)

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(
                    torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(
                    len(self.tokenizer), labels.shape, dtype=torch.long)
                inp[indices_random] = random_words[indices_random].cuda()
                outputs = self.bert_model(inp, masked_lm_labels=labels.long(
                ), attention_mask=(inp != self.tokenizer.pad_token_id).long())
                loss, prediction_scores = outputs[:2]
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)
                print(str(step) + " Loss is :" + str(loss.item()))
                optimizer.step()
                scheduler.step()
            torch.cuda.empty_cache()
        print("LM training done")
        torch.save(self.bert_model.state_dict(), "lm_joke_bert.pth")

    # Use this forward for non-siamese model.
    def forward1(self, *input):
        final_out = []
        input = input[0]
        attn_mask0 = (input[0] != self.tokenizer.pad_token_id).long()
        out_per_seq, _, attention_layer_inps = self.bert_model(
            input[0].long(), attention_mask=attn_mask0)
        out_per_seq = torch.cat((out_per_seq, attention_layer_inps[11]), 2)
        pos = input[0].clone().detach().cpu()
        for (i, loc) in enumerate(input[1]):
            # +1 is to ensure that the symbol token is not considered
            entity1 = torch.mean(out_per_seq[i, loc[0]+1:loc[1]], 0)
            entity2 = torch.mean(out_per_seq[i, loc[2] + 1:loc[3]], 0)
            # Limit attention to original sentence for entity1 and edited sentence for entity2
            imp_seq1 = torch.cat((out_per_seq[i, 0:loc[0] + 1], out_per_seq[i, loc[1]:np.where(
                pos[i].numpy() == self.tokenizer.sep_token_id)[0][0]]), 0)
            imp_seq2 = torch.cat((out_per_seq[i, np.where(pos[i].numpy(
            ) == self.tokenizer.sep_token_id)[0][1]:loc[2] + 1], out_per_seq[i, loc[3]:]), 0)
            _, attention_score = self.attention(
                entity2.unsqueeze(0).unsqueeze(0), imp_seq2.unsqueeze(0))
            sent_attn2 = torch.sum(attention_score.squeeze(
                0).expand(768 * 2, -1).t() * imp_seq2, 0)
            _, attention_score = self.attention(
                entity1.unsqueeze(0).unsqueeze(0), imp_seq1.unsqueeze(0))
            sent_attn1 = torch.sum(attention_score.squeeze(
                0).expand(768 * 2, -1).t() * imp_seq1, 0)
            #attn_diff = torch.abs(sent_attn2-sent_attn1)
            sent_out = self.prelu(self.linear_reg1(
                torch.cat((sent_attn2, sent_attn1, out_per_seq[i, 0], entity2), 0)))
            out = self.final_linear(sent_out)
            final_out.append(out)
        #out = self.final_linear(torch.cat((out_per_seq[:, 0, :],entity_diff), 1))

        return torch.stack(final_out)

    def forward(self, *input):
        '''
        :param input: input[0] is the sentence, input[1] are the entity locations , input[2] is the ground truth
        :return: Scores for each class
        '''

        print(input)

        final_scores = []
        input = input[0]
        attn_mask0 = (input[0] != self.tokenizer.pad_token_id).long()
        output_per_seq1, _, attention_layer_inps = self.bert_model(input[0].long(), attention_mask=attn_mask0)
        output_per_seq1 = torch.cat((output_per_seq1, attention_layer_inps[11]), 2)
        attn_mask1 = (input[1] != self.tokenizer.pad_token_id).long()
        output_per_seq2, _, attention_layer_inps = self.bert_model(input[1].long(), attention_mask=attn_mask1)
        output_per_seq2 = torch.cat((output_per_seq2, attention_layer_inps[11]), 2)
        '''
        Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
        '''
        for (i, loc) in enumerate(input[2]):
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

    def train_non_siamese(self):
        if torch.cuda.is_available():
            self.cuda()
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.001)

        loss = nn.MSELoss()
        train_dataloader = get_sent_emb_dataloaders_bert(
            self.train_file_path, 'train', self.train_batch_size)

        val_dataloader = get_sent_emb_dataloaders_bert(
            self.dev_file_path, "val", self.train_batch_size)
        best_loss = sys.maxsize
        best_accuracy = -sys.maxsize
        steps = 0
        pred_scores = []
        gt_scores = []
        print(f"Pad token is {self.tokenizer.pad_token}")
        for epoch in range(self.epochs):
            steps += 1
            if epoch == 0:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 140, 1400)
            total_prev_loss = 0
            for (batch_num, batch) in enumerate(train_dataloader):
                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input1 = batch[0].cuda()
                    locs = batch[1].cuda()
                    gt = batch[2].cuda()
                else:
                    input1 = batch[0]
                    locs = batch[1]
                    gt = batch[2]

                loss_val = 0
                self.bert_model.train()
                self.attention.train()
                self.linear_reg1.train()
                self.prelu.train()
                self.final_linear.train()

                # Clear gradients
                optimizer.zero_grad()
                final_scores = self.forward1((input1, locs))
                loss_val += loss(final_scores.squeeze(1), gt.float())

                # Compute gradients
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                total_prev_loss += loss_val.item()
                print("Loss for batch" + str(batch_num) +
                      ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()
                scheduler.step()

            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                pred_scores = []
                gt_scores = []
                predictions = []
                ground_truth = []
                self.bert_model.eval()
                self.attention.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
                self.prelu.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input1 = val_batch[0].cuda()
                        locs = val_batch[1].cuda()
                        gt = val_batch[2].cuda()
                    else:
                        input1 = val_batch[0]
                        locs = val_batch[1]
                        gt = val_batch[2]

                    final_scores = self.forward1((input1, locs))
                    pred_scores.extend(final_scores.cpu().detach().squeeze(1))
                    gt_scores.extend(gt.cpu().detach())

                    mse_loss += mean_squared_error(gt.cpu().detach(),
                                                   final_scores.cpu().detach().squeeze(1))

                print(
                    f"Validation Loss is {np.sqrt(mean_squared_error(gt_scores,pred_scores))}")

                if mse_loss < best_loss:
                    torch.save(self.state_dict(), "model_1_" +
                               str(epoch) + ".pth")
                    best_loss = mse_loss

    def train(self, mode=True):
        if torch.cuda.is_available():
            self.cuda()
        #self.bert_model = self.bert_model.roberta
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.001)

        loss = nn.MSELoss()
        train_dataloader = get_dataloaders_bert(
            self.train_file_path, self.model_type, 'train', self.train_batch_size)

        val_dataloader = get_dataloaders_bert(
            self.dev_file_path, self.model_type, "val", self.train_batch_size)
        best_loss = sys.maxsize
        best_accuracy = -sys.maxsize
        steps = 0
        pred_scores = []
        gt_scores = []
        print(f"Pad token is {self.tokenizer.pad_token}")
        for epoch in range(self.epochs):
            steps += 1
            if epoch == 0:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 140, 1400)
            total_prev_loss = 0
            for (batch_num, batch) in enumerate(train_dataloader):
                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input1 = batch[0].cuda()
                    input2 = batch[1].cuda()
                    locs = batch[2].cuda()
                    gt = batch[3].cuda()
                else:
                    input1 = batch[0]
                    input2 = batch[1]
                    locs = batch[2]
                    gt = batch[3]

                loss_val = 0
                self.bert_model.train()
                self.attention.train()
                self.linear_reg1.train()
                self.prelu.train()
                self.final_linear.train()

                # Clear gradients
                optimizer.zero_grad()
                # TODO Change forward arguments
                final_scores = self.forward((input1, input2, locs))
                loss_val += loss(final_scores.squeeze(1), gt.float())

                # Compute gradients
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                total_prev_loss += loss_val.item()
                print("Loss for batch" + str(batch_num) +
                      ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()
                scheduler.step()

            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                pred_scores = []
                gt_scores = []
                predictions = []
                ground_truth = []
                self.bert_model.eval()
                self.attention.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
                self.prelu.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input1 = val_batch[0].cuda()
                        input2 = val_batch[1].cuda()
                        locs = val_batch[2].cuda()
                        gt = val_batch[3].cuda()
                    else:
                        input1 = val_batch[0]
                        input2 = val_batch[1]
                        locs = val_batch[2]
                        gt = val_batch[3]

                    final_scores = self.forward((input1, input2, locs))
                    pred_scores.extend(final_scores.cpu().detach().squeeze(1))
                    gt_scores.extend(gt.cpu().detach())

                    mse_loss += mean_squared_error(gt.cpu().detach(),
                                                   final_scores.cpu().detach().squeeze(1))

                print(
                    f"Validation Loss is {np.sqrt(mean_squared_error(gt_scores,pred_scores))}")

                if mse_loss < best_loss:
                    torch.save(self.state_dict(), "model_1_" +
                               str(epoch) + ".pth")
                    best_loss = mse_loss

    def predict(self, model_path=None):
        '''
        This function predicts the classes on a test set and outputs a csv file containing the id and predicted class
        :param model_path: Path of the model to be loaded if not the current model is used.
        :return:
        '''

        gts = []
        preds = []
        if torch.cuda.is_available():
            self.cuda()
        if model_path:
            self.load_state_dict(torch.load(model_path))
        test_dataloader = get_sent_emb_dataloaders_bert(
            self.test_file_path, self.model_type, "val")
        self.bert_model.eval()
        self.linear_reg1.eval()
        self.final_linear.eval()
        self.prelu.eval()
        self.attention.eval()
        with torch.no_grad():
            with open("task-1-output.csv", "w+") as f:
                f.writelines("id,pred\n")
                for ind, batch in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        input1 = batch[0].cuda()
                        input2 = batch[1].cuda()
                        locs = batch[2].cuda()
                        id = batch[4].cuda()
                        gt = batch[3].cuda()
                    else:
                        input1 = batch[0]
                        input2 = batch[1]
                        locs = batch[2]
                    final_scores_1 = self.forward((input1, input2, locs))
                    preds.extend(final_scores_1.cpu().detach().squeeze(1))
                    gts.extend(gt.cpu().detach())
                    for cnt, pred in enumerate(final_scores_1):
                        f.writelines(str(id[cnt].item()) +
                                     "," + str(pred.item()) + "\n")

                print(
                    f"Test score is {np.sqrt(mean_squared_error(gts,preds))}")
