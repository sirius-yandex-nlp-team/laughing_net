from laughing_net.config import params
from laughing_net.utils.get_paths import get_data_paths
from laughing_net.models.FunBERT.model import FunBERT


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

def train(task, model_type):
    data_paths = get_data_paths(task)

    model = FunBERT(data_paths['train'],
                    data_paths['dev'],
                    data_paths['test'],
                    data_paths['lm'],
                    params.data.train_bathch_size,
                    params.data.test_bathch_size,
                    params.models.FunBERT.learning_rate,
                    'lm_joke_bert.pth',
                    params.models.FunBERT.epochs,
                    None,
                    'model_2.pth',
                    model_type)
    #obj.bert_model = obj.bert_model.roberta
    # obj.bert_model.load_state_dict(torch.load('lm_joke_bert.pth'))
    model.train()


if __name__ == "__main__":
    train(task=1, model_type='bert')
