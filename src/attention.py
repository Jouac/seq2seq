import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size//self.heads

        assert (self.head_dim * self.heads == self.embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * self.heads, self.embed_size)

    def forward(self,query,key,value,mask):
        N = query.shape[0]
        q_len,k_len,v_len = query.shape[1],key.shape[1],value.shape[1]

        # Split embedding into self.heads pieces
        query = self.queries(query.reshape(N,q_len,self.heads,self.head_dim))
        key = self.keys(key.reshape(N,k_len,self.heads,self.head_dim))
        value = self.values(value.reshape(N,v_len,self.heads,self.head_dim))

        energy = torch.einsum("nqhd,nkhd->nhqk",[query,key])
        # query shape: (N, q_len, heads, head_dim)
        # key shape: (N, k_len, heads, head_dim)
        # enerygy shape: (N,heads, q_len, k_len)
        # Normally speaking, we need torch.bmm

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # dim = 3 means we softmax (N,heads, q_len, k_len) in the k_len
        attention = torch.softmax(energy / (self.embed_size**(1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd",[attention,value]).reshape(
            N,q_len,self.heads * self.head_dim
        )
        # attention shape: (N,heads, q_len, k_len)
        # value shape: (N, v_len, heads, head_dim)
        # ->: (N,q_len, heads, head_dim)
        # N,q_len,heads * head_dim

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attension = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size,embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask):
        attention = self.attension(query,key,value,mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out





class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length

    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N,seq_len = x.shape
        position = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,value,key,src_mask,trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query,key,value,src_mask)

        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_out,src_mask,trg_mask):
        N,seq_len = x.shape
        position = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            x = layer(x,enc_out,enc_out,src_mask,trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size = 256,
                 num_layers = 6,
                 forward_expansion = 4,
                 heads = 8,
                 dropout = 0,
                 device = 'cpu',
                 max_length = 100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N,1,1,src_len)
        print(src_mask)
        return src_mask.to(self.device)

    def make_trg_mask(self,trg):
        N,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
            N,1,trg_len,trg_len
        )
        print('trg_mask',trg_mask,trg_mask.shape)
        return trg_mask.to(self.device)

    def forward(self,src,trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg,enc_src,src_mask,trg_mask)
        return out

 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1,2,0],[1,3,2]]).to(device)
    trg = torch.tensor([[3,1,1,0],[2,1,2,3]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 4
    trg_vocab_size = 4
    model = Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)

    out = model(x,trg[:,:-1])
    # print(out)
    # print(out.shape)



