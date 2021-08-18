import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)
        
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        #embedded = [src sent len, batch size, emb dim]
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        return output, hidden, cell
    
    

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, num_channels_conv=512, kernel_size=5):
        super(ConvEncoder, self).__init__()

        self.word_embedding = nn.Embedding(input_dim, emb_dim)
        self.linear = nn.Linear(in_features=emb_dim, out_features=hid_dim)
        self.num_layers = n_layers
        self.dropout = dropout
        self.conv_num = 3

        self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size=3,
                                      padding=2) for _ in range(self.conv_num)])
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, sentence_as_wordids, position_ids=None):
        word_embedding = self.word_embedding(sentence_as_wordids)
        embedded = None

        if position_ids != None:
            position_embedding = torch.zeros(position_ids.size()[0], word_embedding.size()[1], position_ids.size()[1]).to(device)
            for i in range(word_embedding.size()[1]):
                position_embedding[:,i,:] = position_ids
        
            embedded = F.dropout(word_embedding + position_embedding, self.dropout, self.training)
        else :
            embedded = F.dropout(word_embedding, self.dropout, self.training)
        embedded = self.linear(embedded)


        cnn = embedded.permute(0, 2, 1)

        for i, layer in enumerate(self.conv):
            cnn = F.tanh(layer(cnn)+cnn)  

        return torch.unsqueeze(torch.mean(cnn.permute(0, 2, 1), 0), 0).contiguous(), cnn.permute(0, 2, 1).contiguous()

    
def positional_encoding(seq_len, dim_model, device):
    pos = torch.arange(seq_len, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase)).squeeze(0)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
            # <YOUR CODE HERE>
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
            # <YOUR CODE HERE>
        
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
            # <YOUR CODE HERE>
        
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
        
    def forward(self, input, hidden, cell, encoder_outputs=None):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))# <YOUR CODE HERE>
        
        #embedded = [1, batch size, emb dim]

        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, encoder_outputs):
        hidden_state = self.linear(hidden_state.permute(1, 0, 2))
        attention_weights = F.softmax(torch.bmm(hidden_state, encoder_outputs.permute(1, 2, 0)), -1)
        attention_vector = torch.bmm(attention_weights, encoder_outputs.permute(1, 0, 2))
        return attention_vector.permute(1, 0, 2).contiguous()


class AttentionDecoder(Decoder):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(AttentionDecoder, self).__init__(output_dim, emb_dim,
                                               hid_dim, n_layers, dropout)
        self.attention = Attention(hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        attention_vector = self.attention(hidden, encoder_outputs)
        hidden_with_attention = attention_vector + hidden
        prediction = self.out(hidden_with_attention[-1])

        return prediction, hidden_with_attention, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, cnn_flag=None):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.cnn_flag = cnn_flag
        self.emb_dim = encoder.emb_dim
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs = None
        hidden = None
        cell = None

        if self.cnn_flag == None:
            encoder_outputs, hidden, cell = self.encoder(src)
        else:
            pos_ids = positional_encoding(src.size()[0], self.emb_dim, self.device)
            #pos_ids = None 
            hidden, encoder_outputs = self.encoder(src, pos_ids)
            cell = torch.zeros_like(hidden)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
