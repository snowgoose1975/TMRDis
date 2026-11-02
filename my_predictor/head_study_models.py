"""Staging copy for installation in the canonical WSL repository only."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from my_predictor.model import ResidueMLP, ResidueLinear


class StudyHead(nn.Module):
    def __init__(self, dimension, kind, width=256, dilations=(1,2,4), dropout=.1):
        super().__init__()
        self.kind, self.width = kind, width
        self.project = nn.Linear(dimension, width)
        self.activation, self.drop = nn.GELU(), nn.Dropout(dropout)
        self.output = nn.Linear(width * (2 if kind in ('bigru','bilstm') else 1), 1)
        if kind in ('deep_mlp','residual_mlp'):
            self.blocks = nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
        elif kind in ('plain_cnn','residual_cnn','dilated_cnn','depthwise_cnn'):
            rates = dilations if kind=='dilated_cnn' else (1,1,1)
            self.blocks = nn.ModuleList()
            for rate in rates:
                def conv():
                    if kind=='depthwise_cnn':
                        return nn.Sequential(nn.Conv1d(width,width,5,padding=2*rate,dilation=rate,groups=width),nn.Conv1d(width,width,1))
                    return nn.Conv1d(width,width,5,padding=2*rate,dilation=rate)
                self.blocks.append(nn.ModuleList([conv(),conv()]))
        elif kind=='multiscale_cnn':
            self.blocks=nn.ModuleList([nn.Conv1d(width,width,k,padding=k//2) for k in (3,5,9)])
            self.fuse=nn.Linear(3*width,width)
        elif kind in ('bigru','bilstm'):
            self.rnn=(nn.GRU if kind=='bigru' else nn.LSTM)(width,width,batch_first=True,bidirectional=True)
        elif kind=='window_transformer':
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,4*width,dropout,batch_first=True,activation='gelu'),2,enable_nested_tensor=False)
            self.position=nn.Parameter(torch.zeros(1,256,width))
            nn.init.normal_(self.position,std=.02)
        else:raise ValueError(kind)

    def forward(self,x,lengths=None):
        if lengths is None:lengths=torch.full((len(x),),x.shape[1],dtype=torch.long)
        valid=torch.arange(x.shape[1],device=x.device)[None,:] < lengths.to(x.device)[:,None]
        def mask(v):return v.masked_fill(~valid[:,:,None],0)
        v=mask(self.project(mask(x)))
        if self.kind in ('deep_mlp','residual_mlp'):
            v=mask(self.drop(self.activation(v)))
            for block in self.blocks:
                out=self.drop(self.activation(block(v)))
                v=mask(v+out if self.kind=='residual_mlp' else out)
        elif self.kind in ('plain_cnn','residual_cnn','dilated_cnn','depthwise_cnn'):
            for c1,c2 in self.blocks:
                z=mask(self.activation(c1(v.transpose(1,2)).transpose(1,2)))
                z=mask(self.drop(c2(z.transpose(1,2)).transpose(1,2)))
                v=mask(self.activation(z if self.kind=='plain_cnn' else z+v))
        elif self.kind=='multiscale_cnn':
            z=torch.cat([mask(self.activation(c(v.transpose(1,2)).transpose(1,2))) for c in self.blocks],-1)
            v=mask(self.drop(self.activation(self.fuse(z))))
        elif self.kind in ('bigru','bilstm'):
            packed=pack_padded_sequence(v,lengths.cpu(),batch_first=True,enforce_sorted=False)
            v,_=self.rnn(packed)
            v,_=pad_packed_sequence(v,batch_first=True,total_length=x.shape[1])
            v=self.drop(v)
        elif self.kind=='window_transformer':
            # Independent fixed windows bound attention memory; not global attention.
            outputs=[]
            for row,n in enumerate(lengths.tolist()):
                chunks=[]
                for start in range(0,n,256):
                    z=v[row:row+1,start:min(start+256,n)]
                    chunks.append(self.encoder(z+self.position[:,:z.shape[1]]))
                out=torch.cat(chunks,1)
                outputs.append(torch.nn.functional.pad(out,(0,0,0,x.shape[1]-n)))
            v=torch.cat(outputs,0)
        return self.output(v).squeeze(-1).masked_fill(~valid,0)


KINDS=('linear','mlp','deep_mlp','residual_mlp','plain_cnn','residual_cnn',
       'dilated_cnn','multiscale_cnn','depthwise_cnn','bigru','bilstm','window_transformer')


def build(dimension,spec):
    kind=spec['head'];width=spec.get('width',256)
    if kind=='linear':return ResidueLinear(dimension)
    if kind=='mlp':return ResidueMLP(dimension,width,.1)
    return StudyHead(dimension,kind,width,tuple(spec.get('dilations',(1,2,4))))
