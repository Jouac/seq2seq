import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
# from utils import translate_sentence,bleu,save_checkpoint,load_checkpoint


spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')
# print(spacy_eng.__dict__)


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
# text = 'I come from China, I love Chinese culture.'
# res = [tok.text for tok in spacy_eng.tokenizer(text)]
# print(res)

english = Field(tokenize = tokenizer_eng,lower=True,init_token='<sos>',eos_token='<eos>')
german = Field(tokenize = tokenizer_ger,lower=True,init_token='<sos>',eos_token='<eos>')

train_data,val_data,test_data = Multi30k.splits(exts =('.de','.en'),fields=(german,english))

german.build_vocab(train_data,max_size = 10000, min_freq=2)
english.build_vocab(train_data,max_size = 10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,dropout_rate):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout = dropout_rate)

    def forward(self,x):
        # x : [seq_len, batch_size]
        embedding = self.dropout(self.embedding(x))
        # embedding : [seq_len, batch_size, embedding_size]
        outputs,(hidden,cell) = self.rnn(embedding)
        return hidden,cell

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,dropout_rate):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x,hidden,cell):

        # x : [N] but we want x : [1,N] , every timestep, it eat a single word
        x = x.unsqueeze(0)
        # x : [1,N]
        embedding = self.dropout(self.embedding(x))
        # embedding: [1, N , embedding_size]
        outputs,(hidden,cell) = self.rnn(embedding,(hidden,cell))
        # output: [1, N, hidden_size]
        prediction = self.fc(outputs)
        # prediction: [1, N, output_size]
        prediction = prediction.squeeze(0)

        return prediction,hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,source,target,teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        target_vocab_size = len(english.vocab)
        # ouputs_vocab_size = len(german.vocab)
        hidden,cell = self.encoder(source)

        outputs = torch.zeros(target_len,batch_size,target_vocab_size)

        # Grab start token
        x = target[0]

        for t in range(1,target_len):
            output, hidden, cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            # (N, english_vocab_size)
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


### Now we're ready to do the training
# Training Hyperparameters

NUM_EPOCHES = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

# Model Hyperparameters
LOAD_MODEL = False
INPUT_SIZE_ENCODER = len(german.vocab)
INPUT_SIZE_DECODER = len(english.vocab)
OUTPUTS_SIZE =  len(english.vocab)

ENCODER_EMBEDDING_SIZE = 300
DECODER_EMBEDDING_SIZE = 300

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
ENCODER_DROPOUT = 0.5
DECODER_DROPOUT = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data,val_data,test_data),batch_size = BATCH_SIZE, sort_within_batch = True,sort_key = lambda x : len(x.src))

encoder_net = Encoder(INPUT_SIZE_ENCODER,ENCODER_EMBEDDING_SIZE,HIDDEN_SIZE,NUM_LAYERS,ENCODER_DROPOUT)
decoder_net = Decoder(INPUT_SIZE_DECODER,DECODER_EMBEDDING_SIZE,HIDDEN_SIZE,OUTPUTS_SIZE,NUM_LAYERS,DECODER_DROPOUT)
model = Seq2Seq(encoder_net,decoder_net)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
#
# if LOAD_MODEL:
#     load_checkpoint(torch.load('my_checkpoint.pth.ptor'),model,optimizer)

# sentence = 'ein boot mit mehreren mannern darouf wird von einem groben pferdegespann ans ufer gezogen.'

for epoch in range(NUM_EPOCHES):
    # print(f'Epoch {epoch} / {NUM_EPOCHES}')
    checkpoint = {'state_dict':model.state_dict(),
                  'optimizer':optimizer.state_dict()}
    # save_checkpoint(checkpoint)

    model.train()
    for batch_idx, batch in enumerate(train_iterator):
        source = batch.src
        target = batch.trg

        output = model(source,target)
        # output : [target_len,batch_size,target_vocab_size]

        output = output[1:].reshape(-1,output.shape[2]) # the first is the start token, so we remove it.
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1)
        optimizer.step()

        writer.add_scalar('Training loss',loss,global_step=step)
        step += 1
        print(f'Epoch:{epoch} / {NUM_EPOCHES},loss:{loss.item():.3f}')








score = bleu(test_data,model,german,english)



print('done')
