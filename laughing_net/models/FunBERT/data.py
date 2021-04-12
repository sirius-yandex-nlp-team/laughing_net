import torch
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer,DistilBertTokenizer,RobertaTokenizer


# Get the tokenized input and the locations of the focus words.
def tokenize_bert(X: list, org: bool, tokenizer_type='roberta'):
    '''
    This function tokenizes the input sentences and returns a vectorized representation of them and the location
    of each entity in the sentence.

    :param X: List of all input sentences
    :return: A vectorized list representation of the sentence and a numpy array containing the locations of each entity. First two
    values in  a row belong to entity1 and the next two values belong to entity2.
    '''

    # Add the SOS and EOS tokens.
    # TODO: Replace fullstops with [SEP]
    if tokenizer_type == 'roberta':
      tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
      sentences = ["<s> " + sentence + " </s>" for sentence in X]
    elif tokenizer_type == 'bert':
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      sentences = ["[CLS] " + sentence + " [SEP]" for sentence in X]
    elif tokenizer_type == 'distilbert':
      tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
      sentences = ["[CLS] " + sentence + " [SEP]" for sentence in X]
    

    # Tokenize and vectorize
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]
    print(tokenizer_type)
    print(tokenized_text[0])
    print(tokenizer.pad_token)
    print(tokenizer.pad_token_id)

    # MAX_SEQ_LEN
    MAX_LEN = 50
    #Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',value=tokenizer.pad_token_id)
    print(X[0])
    
    if tokenizer_type != 'roberta':
      if org:
          entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '<'] for sent in tokenized_text])
      else:
          entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '^'] for sent in tokenized_text])
    
    else:
      # Find the locations of each entity and store them
      if org:
          entity_locs = np.asarray([[i for i, s in enumerate(sent) if '<' in s and len(s)==2] for sent in tokenized_text])
      else:
          entity_locs = np.asarray([[i for i, s in enumerate(sent) if '^' in s and len(s)==2] for sent in tokenized_text])
    print(entity_locs[0]) 
    return X,entity_locs


# This function is to get the dataloaders to repeat the experiment of using a Non-Siamese network.
def get_sent_emb_dataloaders_bert(file_path: str, mode='train', train_batch_size=64, test_batch_size=64, model=None):
    
    df = pd.read_csv(file_path, sep=",")
    # Get the additional data.
    if mode == 'train':
        df1 = pd.read_csv(file_path[:-4] + "_funlines.csv", sep=",")
        df = pd.concat([df, df1], ignore_index=True)
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"", "") for sent in X]
    # Replaced word
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">") + 1])
    replaced_clean = [x.replace("<", "").replace("/>", "") for x in replaced]
    if mode != 'test':
        y = df['meanGrade'].values
    edit = df['edit']
    # Substitute the edit word in the place of the replaced word add the required demarcation tokens.
    X2 = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
    X1 = [sent.replace("<", "< ").replace("/>", " <") for i, sent in enumerate(X)]
    X, entity_locs = tokenize_roberta_sent(X1, X2)

    if mode == "train":

        train1_inputs = torch.tensor(X)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(entity_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        # train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train_entity_locs, train_labels)
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

        # validation_data = TensorDataset(validation1_inputs,validation2_inputs, validation_entity_locs, validation_word2vec_locs, validation_labels)
        return train_dataloader

    if mode == "val":
        test1_input = torch.tensor(X)
        y = torch.tensor(y)
        train_entity_locs = torch.tensor(entity_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, train_entity_locs,y, id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

    if mode == "test":
        test1_input = torch.tensor(X)
        test2_input = torch.tensor(sent_emb)

        train_entity_locs = torch.tensor(entity_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs, id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


# Get the data loader for language modeling assuming that the model is roberta.
def get_bert_lm_dataloader(file_path : str,batch_size = 16):
    jokes_df = pd.read_csv(file_path)
    jokes = jokes_df['Joke']
    jokes = "<s> " + jokes + " </s>"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    X = [tokenizer.encode(sent,add_special_tokens=False) for sent in jokes]
    MAX_LEN = max([len(sent) for sent in X])
    print(MAX_LEN)
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',tokenizer.pad_token_id)
    dataset = TensorDataset(torch.tensor(X))
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,pin_memory=True)
    return data_loader


# This function is to get the data loaders for the Siamese architecture.
def get_dataloaders_bert(file_path: str, model_type, mode="train",train_batch_size=64,test_batch_size=64):

    '''
    This function creates pytorch dataloaders for fast and easy iteration over the dataset.

    :param file_path: Path of the file containing train/test data
    :param mode: Test mode or Train mode
    :param train_batch_size: Size of the batch during training
    :param test_batch_size: Size of the batch during testing
    :return: Dataloaders
    '''

    # Read the data,tokenize and vectorize
    df = pd.read_csv(file_path, sep=",")
    if mode=='train':
        df1 = pd.read_csv(file_path[:-4]+"_funlines.csv",sep=",")
        df = pd.concat([df,df1],ignore_index=True)
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"","") for sent in X]
    
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">")+1])
    replaced_clean = [x.replace("<","").replace("/>","") for x in replaced]
    if mode!='test':
        y = df['meanGrade'].values
    edit = df['edit']
    X2 = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
    X1 = [sent.replace("<", "< ").replace("/>", " <") for i, sent in enumerate(X)]
    X1,e1_locs = tokenize_bert(X1,True,model_type)
    X2,e2_locs = tokenize_bert(X2,False,model_type)

    replacement_locs = np.concatenate((e1_locs, e2_locs), 1)
    
    if mode == "train":


        train1_inputs = torch.tensor(X1)
        train2_inputs = torch.tensor(X2)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(replacement_locs)
        
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        #train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        return  train_dataloader

    if mode == "val":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)

        train_entity_locs = torch.tensor(replacement_locs)
        y = torch.tensor(y)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs,y,id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
        return test_data_loader

    if mode == "test":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)

        train_entity_locs = torch.tensor(replacement_locs)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs,id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


# Tokenizer function to be used by non-siamese architecture assuming transformer model is Roberta.
def tokenize_roberta_sent(X1: list, X2 : list ):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    sentences = ["<s> " + X1[i] + " </s></s> " + X2[i] + " </s>" for i in range(len(X1))]
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    print(tokenized_text[0])
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]
    print(X[0])
    #sent_emb = [[0 if i<sentence.index(102) else 1 for i  in range(len(sentence)) ] for sentence in X]
    MAX_LEN = max([len(x) for x in X])+1
    print(MAX_LEN)
    # Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',tokenizer.pad_token_id)
    #sent_emb = pad_sequences(sent_emb,MAX_LEN,'long','post','post',1)
    # Find the locations of each entity and store them
    entity_locs1 = np.asarray(
            [[i for i, s in enumerate(sent) if '<' in s and len(s) == 2] for sent in tokenized_text])
    entity_locs2 = np.asarray([[i for i, s in enumerate(sent) if '^' in s and len(s) == 2] for sent in tokenized_text])

    return X,np.concatenate((entity_locs1, entity_locs2), 1)