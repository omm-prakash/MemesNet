import torch
import torch.nn as nn

class Embedding(nn.Module):
        def __init__(self, d_model:int, vocab_size:int) -> None:
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size,d_model)

        def forward(self, x):
                return self.embedding(x)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.int64, requires_grad=False)) # (batch, seq_len, d_model)
        
class PositionEmbedding(nn.Module):
        def __init__(self, d_model:int, dropout:float, seq_len:int) -> None:
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.d_model = d_model

                pos_embedding = torch.empty(seq_len, d_model) # (seq_len, d_model)
                pos = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # (seq_len,1)
                denom = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)*(-torch.log(torch.tensor(10000.0))/d_model)) # (d_model/2)
                pos_embedding[:,0::2] = torch.sin(pos*denom) # (seq_len, d_model/2)
                pos_embedding[:,1::2] = torch.cos(pos*denom) # (seq_len, d_model/2)

                self.pos_embedding = pos_embedding.unsqueeze(0) # (1, seq_len, d_model)
                if not hasattr(self, 'pos_embedding'):
                        self.register_buffer('pos_embedding', pos_embedding)

        def forward(self, x):
                x = x+(self.pos_embedding[:,:x.shape[1],:]).requires_grad_(False)
                return self.dropout(x) # (batch, seq_len, d_model)

class MultiHeadAttention(nn.Module):
        def __init__(self, d_model:int, head:int) -> None:
                super().__init__()
                self.head = head
                self.d_model = d_model
                self.softmax = nn.Softmax(dim=-1)
                self.Wq = nn.Linear(d_model,d_model)
                self.Wv = nn.Linear(d_model,d_model)
                self.Wk = nn.Linear(d_model,d_model)
                self.Wo = nn.Linear(d_model,d_model)

        def forward(self,query,value,key,mask=None):
                Q = self.Wq(query) # (batch, seq_len, d_model)
                V = self.Wv(value) # (batch, seq_len, d_model)
                K = self.Wk(key) # (batch, seq_len, d_model)

                kd = self.d_model//self.head
                Q = Q.view(Q.shape[0],Q.shape[1],self.head,kd).transpose(1,2) # (batch, head, seq_len, kd)
                V = V.view(V.shape[0],V.shape[1],self.head,kd).transpose(1,2) # (batch, head, seq_len, kd)
                K = K.view(K.shape[0],K.shape[1],self.head,kd).transpose(1,2) # (batch, head, seq_len, kd)
                
                score = (Q @ K.transpose(-1,-2))/torch.sqrt(torch.tensor(kd,dtype=torch.int64,requires_grad=False)) # (batch, head, seq_len, seq_len)
                if mask is not None:
                        score.masked_fill_(mask==0,-1*torch.inf)
                score = self.softmax(score) # (batch, head, seq_len, seq_len)
                
                x = score @ V # (batch, head, seq_len, kd)                
                x = x.transpose(1,2).contingious().view(x.shape[0],-1,self.d_model) # (batch, seq_len, d_model)
                return self.Wo(x) # (batch, seq_len, d_model)

class FeedForward(nn.Module):
        def __init__(self, d_model:int) -> None:
                super().__init__()
                self.relu = nn.ReLU()
                self.linear1 = nn.Linear(d_model,d_model)
                self.linear2 = nn.Linear(d_model,d_model)
        
        def forward(self, x):
                return self.linear2(self.relu(self.linear1(x))) # (batch, seq_len, d_model)

class NormalizeLayer(nn.Module):
        def __init__(self, eps:float=10**-6) -> None:
                super().__init__()
                self.eps = eps
                self.alpha = nn.parameter.Parameter(torch.ones(1))
                self.bias = nn.parameter.Parameter(torch.zeros(1))

        def forward(self, x):
                mean = torch.mean(x,dim=-1,keepdim=True) # (batch, seq_len)
                std = torch.std(x,dim=-1,keepdim=True) # (batch, seq_len)

                return self.alpha*((x-mean)/(std+self.eps))+self.bias # (batch, seq_len, d_model)

class ResidualConnection(nn.Module):
        def __init__(self, drop:float) -> None:
                super().__init__()
                self.norm = NormalizeLayer()
                self.dropout = nn.Dropout(drop)

        def forward(self,x,sublayer):
                return self.norm(x+self.dropout(sublayer(x))) # (batch, seq_len, d_model)

class SingleModalityEncoderBlock(nn.Module):
        def __init__(self, drop:float, d_model:int, head:int) -> None:
                super().__init__()
                self.residual1 = ResidualConnection(drop)
                self.residual2 = ResidualConnection(drop)
                self.attention = MultiHeadAttention(d_model,head)
                self.linear = FeedForward(d_model)

        def forward(self, x, mask):
                x = self.residual1(x,lambda x: self.attention(x,x,x,mask))
                x = self.residual2(x,lambda x: self.linear(x))
                return x # (batch, seq_len, d_model)
        
class CrossModalityEncoderBlock(nn.Module):
        def __init__(self, drop:float, d_model:int, head:int) -> None:
                super().__init__()
                self.encoder1 = SingleModalityEncoderBlock(drop,d_model,head)
                self.encoder2 = SingleModalityEncoderBlock(drop,d_model,head)
                self.cross_attention1 = MultiHeadAttention(d_model,head)
                self.cross_attention2 = MultiHeadAttention(d_model,head)
                self.residual1 = ResidualConnection(drop)
                self.residual2 = ResidualConnection(drop)
                
        def forward(self, x, y, mask, drop_encoder_y=False):
                x_ = self.residual1(x,lambda x: self.cross_attention1(x,y,y,mask)).detach()
                x_ = self.encoder1(x_,mask)

                if not drop_encoder_y:
                        y = self.residual2(y,lambda y: self.cross_attention1(y,x,x,mask))
                        y = self.encoder2(y,mask)
                return x_,y # (batch, seq_len, d_model), (batch, seq_len, d_model)
        
class SingleModalityEncoder(nn.Module):
        def __init__(self, N:int, drop:float, d_model:int, head:int) -> None:
                super().__init__()
                self.n = N
                self.single_modality_encoder = nn.ModuleList([SingleModalityEncoderBlock(drop,d_model,head) for _ in range(N)])

        def forward(self, x, mask):
                for i in range(self.n):
                        x = self.single_modality_encoder[i](x,mask)
                return x # (batch, seq_len, d_model)
        
class CrossModalityEncoder(nn.Module):
        def __init__(self, N:int, drop:float, d_model:int, head:int) -> None:
                super().__init__()
                self.n = N
                self.cross_modality_encoder = nn.ModuleList([CrossModalityEncoderBlock(drop,d_model,head) for _ in range(N)])

        def forward(self, x, y, mask):
                for i in range(self.n-1):
                        x,y = self.cross_modality_encoder[i](x,y,mask)
                x,y = self.cross_modality_encoder[-1](x,y,mask,drop_encoder_y=True)

                return x,y # (batch, seq_len, d_model), (batch, seq_len, d_model)

class Projection(nn.Module):
        def __init__(self, d_model:int, seq_len:int) -> None:
                super().__init__()
                self.linear1 = nn.Linear(d_model, 1)
                self.linear2 = nn.Linear(seq_len, 2)

        def forward(self, x):
                x = self.linear1(x) # (batch, seq_len, 1)
                x = x.squeeze(dim=-1) # (batch, seq_len)
                x = self.linear2(x) # (batch, 2)
                return x # (batch, 2)

class MemesNet(nn.Module):
        def __init__(self, 
                     n_single:int,
                     n_cross:int,
                     d_model:int,
                     vocab_size:int,
                     img_vocab_size:int, 
                     dropout:float, 
                     head:int,
                     seq_len:int) -> None:
                super().__init__()
                self.text_embedding = Embedding(d_model, vocab_size)
                self.pos_embedding = PositionEmbedding(d_model, dropout, seq_len)
                self.img_embedding = Embedding(d_model, img_vocab_size)

                self.img_encoder = SingleModalityEncoder(n_single, dropout, d_model, head) 
                self.text_encoder = SingleModalityEncoder(n_single, dropout, d_model, head)
                self.cross_encoder = CrossModalityEncoder(n_cross, dropout, d_model, head)
                
                self.projection = Projection(d_model, seq_len)

        def forward(self, x, y, mask):
                x = self.text_embedding(x) # (batch, seq_len, d_model)
                x = self.pos_embedding(x) # (batch, seq_len, d_model)
                x = self.text_encoder(x) # (batch, seq_len, d_model)
                
                y = self.img_embedding(y) # (batch, seq_len, d_model)
                y = self.img_encoder(y) # (batch, seq_len, d_model)

                x,_ = self.cross_encoder(x,y,mask) # (batch, seq_len, d_model)
                x = self.projection(x) # (batch, 2)
                return x # (batch, 2)


def memesnet(n_single:int,
             n_cross:int,
             d_model:int,
             vocab_size:int,
             img_vocab_size:int, 
             dropout:float, 
             head:int,
             seq_len:int):
        
        model = MemesNet(n_single,n_cross,d_model,vocab_size,img_vocab_size,dropout,head,seq_len)

        # initilize the model parameters
        for param in model.parameters():
                if param.dim()>1:
                        nn.init.xavier_uniform_(param)

        return model