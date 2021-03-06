import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

from baseline_utils import SYMBOLS, INP_VOCAB, RES_VOCAB, DEVICE, NULL, END, RES_MAX_LEN, resnet_scan
from gnn.gcnn import GGNN
from gnn.util import S2VGraph

class EmbeddingIn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_input = not config.perception

        if self.image_input:
            self.image_encoder = resnet_scan.make_model(n_class=len(SYMBOLS))
        self.n_token = len(SYMBOLS) + 3
        self.embedding = nn.Embedding(self.n_token, config.emb_dim)
        
    def forward(self, src, src_len):
        if self.image_input:
            logits = self.image_encoder(src)
            probs = F.softmax(logits, dim=-1)
            src = torch.matmul(probs, self.embedding.weight[:-3])
        else:
            src = self.embedding(src)

        max_len = src_len.max()
        current = 0
        padded_src = []
        emb_start = self.embedding(torch.tensor([self.n_token - 3]).to(DEVICE))
        emb_end = self.embedding(torch.tensor([self.n_token - 2]).to(DEVICE))
        emb_null = self.embedding(torch.tensor([self.n_token - 1]).to(DEVICE))
        for l in src_len:
            current_input = src[current:current+l]
            current_input = [emb_start, current_input, emb_end] + [emb_null] * (max_len - l) 
            current_input = torch.cat(current_input)
            padded_src.append(current_input)
            current += l
        src = torch.stack(padded_src)
        return src

class RNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        emb_dim = config.emb_dim
        hid_dim = config.hid_dim
        enc_layers = config.enc_layers
        dec_layers = config.dec_layers
        dropout = config.dropout

        self.dec_hid_dim = hid_dim * 2
        self.encoder = nn.GRU(emb_dim, hid_dim, enc_layers, dropout=dropout, bidirectional=True)
        self.decoder = nn.GRU(emb_dim, self.dec_hid_dim, dec_layers, dropout=dropout, bidirectional=False)

        self.embedding_out = nn.Embedding(len(RES_VOCAB), config.emb_dim)
        self.classifier_out = nn.Linear(self.dec_hid_dim, len(RES_VOCAB))

    def forward(self, src, tgt, src_len=None, tgt_len=None):
        _, hidden = self.encoder(src)
        hidden = hidden.view(-1, 2, *hidden.shape[1:])
        hidden = hidden[-1]
        hidden = hidden.transpose(0, 1)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        hidden = torch.stack([hidden] * self.config.dec_layers)

        if self.training:
            tgt = self.embedding_out(tgt)
            output, _ = self.decoder(tgt, hidden)
            output = self.classifier_out(F.relu(output))
        else:
            pred = tgt[0]
            output_list = []
            finish = torch.zeros((src.shape[1])).bool().to(DEVICE)
            while not finish.all() and len(output_list) <= RES_MAX_LEN:
                pred = pred.unsqueeze(0)
                pred = self.embedding_out(pred)
                output, hidden = self.decoder(pred, hidden)
                output = output[0]
                output = self.classifier_out(F.relu(output))
                pred = output.argmax(1)
                pred[finish] = RES_VOCAB.index(NULL)
                finish[pred == RES_VOCAB.index(END)] = True
                output_list.append(output)
            
            output = torch.stack(output_list)
        return output


class SinePositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        seq_len, batch_size = x.shape[:2]
        pos = torch.arange(0, seq_len).unsqueeze(1).to(DEVICE)
        x = x + self.pe(pos)
        return self.dropout(x)

def create_padding_mask(lens):
    # 1, pos is masked and not allowed to attend;
    max_len = max(lens)
    batch_size = len(lens)
    mask = np.ones((batch_size, max_len+2)) # 2 for the START, END token
    for i, l in enumerate(lens):
        mask[i, :l+2] = 0 
    return mask.astype(bool)

def create_padding_mask_tgt(tgt):
    mask = tgt == RES_VOCAB.index(NULL)
    return mask

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nhead = config.nhead
        emb_dim = config.emb_dim
        hid_dim = config.hid_dim
        enc_layers = config.enc_layers
        dec_layers = config.dec_layers
        dropout = config.dropout

        self.d_model = emb_dim
        # self.enc_pos_embedding = SinePositionalEncoding(emb_dim, dropout)
        # self.dec_pos_embedding = SinePositionalEncoding(emb_dim, dropout)
        self.enc_pos_embedding = LearnedPositionalEncoding(emb_dim, dropout)
        self.dec_pos_embedding = LearnedPositionalEncoding(emb_dim, dropout)
        self.transformer = nn.Transformer(emb_dim, nhead, enc_layers, dec_layers, hid_dim, dropout)

        self.embedding_out = nn.Embedding(len(RES_VOCAB), emb_dim)
        self.classifier_out = nn.Linear(emb_dim, len(RES_VOCAB))


    def forward(self, src, tgt, src_len, tgt_len):
        src_padding_mask = torch.from_numpy(create_padding_mask(src_len)).to(DEVICE)
        src = self.enc_pos_embedding(src * math.sqrt(self.d_model))
        if self.training:
            tgt_padding_mask = create_padding_mask_tgt(tgt).transpose(0, 1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(DEVICE)
            tgt_emb = self.embedding_out(tgt)
            tgt_emb = self.dec_pos_embedding(tgt_emb * math.sqrt(self.d_model))
            output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                            src_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask)
            output = self.classifier_out(F.relu(output))
        else:
            tgt = tgt[:1]
            output_list = []
            finish = torch.zeros((src.shape[1])).bool().to(DEVICE)
            while not finish.all() and len(tgt) <= RES_MAX_LEN:
                tgt_padding_mask = create_padding_mask_tgt(tgt).transpose(0, 1)
                tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(DEVICE)
                tgt_emb = self.embedding_out(tgt)
                tgt_emb = self.dec_pos_embedding(tgt_emb * math.sqrt(self.d_model))
                output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                            src_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask)
                output = output[-1]
                output = self.classifier_out(F.relu(output))
                pred = output.argmax(1)
                pred[finish] = RES_VOCAB.index(NULL)
                tgt = torch.cat([tgt, pred.unsqueeze(0)])
                finish[pred == RES_VOCAB.index(END)] = True

                output_list.append(output)
            
            output = torch.stack(output_list)
        return output

def make_graph(src, src_len, head):
    graphs = []
    for nodes, l, h in zip(src, src_len, head):
        nodes = nodes[:l]
        neighbors = [[i] if i != -1 else [] for i in h]
        g = S2VGraph(None, nodes, neighbors)
        graphs.append(g)
    return graphs

class NeuralArithmetic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_in = EmbeddingIn(config)
        if config.seq2seq == 'RNN':
            self.seq2seq = RNNModel(config)
        elif config.seq2seq == 'TRAN':
            self.seq2seq = TransformerModel(config)
        elif config.seq2seq == 'GNN':
            self.seq2seq = GGNN(config)

    
    def forward(self, img, src, tgt, src_len, tgt_len, head=None):
        src = self.embedding_in(src if self.config.perception else img, src_len)
        if self.config.seq2seq == 'GNN':
            src = make_graph(src, src_len, head)
            output = self.seq2seq(src)
            # output = torch.exp(output)
        else:
            output = self.seq2seq(src.transpose(0,1), tgt.transpose(0,1), src_len, tgt_len)
            output = output.transpose(0, 1)
        return output

def make_model(config):
    model = NeuralArithmetic(config)
    return model