import torch
from torch import nn
from . import utils


class Permute(nn.Module):
    def __init__(self, *order):
        super().__init__()
        self.order = order
        
    def forward(self, x):
        return x.permute(self.order)

class Onehot(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def forward(self, x):
        return nn.functional.one_hot(x.type(torch.long), num_classes=self.n).type(torch.float32)

class SRAMP(nn.Module):
    def __init__(self, mode='full', halfseqlen=400, oh=True):
        super().__init__()
        self.mode = mode
        self.halfseqlen = halfseqlen
        
        if oh: self.oh = Onehot(6)
        else: self.oh = oh
        
        self.conv_model = nn.Sequential(
            Permute(0, 2, 1),
            nn.Conv1d(6, utils.conv_len, 3, padding=1),
            nn.Conv1d(utils.conv_len, utils.conv_len, 5, padding=2),
            Permute(0, 2, 1),
        )
        
        self.token_emb = nn.Sequential(
            nn.Linear(6, utils.latent - utils.conv_len - utils.emb_len)
        )
        
        self.genometrans = nn.Sequential(
            nn.Linear(utils.emb_len, utils.emb_len),
        )
        
        self.trans = nn.Sequential(
            nn.TransformerEncoderLayer(utils.latent, 4, 4*utils.latent, dropout=0.2, activation='gelu', batch_first=True, norm_first=True),
        )
        
        self.lstm = nn.GRU(utils.latent, utils.latent // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2) 
        
        self.tail = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(utils.latent + utils.fea_len, utils.latent // 4),
            Permute(0, 2, 1),
            nn.BatchNorm1d(utils.latent // 4),
            Permute(0, 2, 1),
            nn.GELU(),
            nn.Linear(utils.latent // 4, utils.latent // 8),
            Permute(0, 2, 1),
            nn.BatchNorm1d(utils.latent // 8),
            Permute(0, 2, 1),
            nn.GELU(),
            nn.Linear(utils.latent // 8, 2),
            # nn.Sigmoid(),
        )
        
    def forward(self, x, emb, *args):
        if utils.half_length > self.halfseqlen:
            x = x[:, utils.half_length-self.halfseqlen:utils.half_length+self.halfseqlen+1]
            emb = emb[:, utils.half_length-self.halfseqlen:utils.half_length+self.halfseqlen+1]
        
        if self.oh: x = self.oh(x)
        
        te = self.token_emb(x)
        me = self.conv_model(x)
        
        x = torch.cat([te, me], axis=-1)
        
        if self.mode == 'seqonly': emb = emb.zero_()
        elif self.mode == 'genomeonly': x = x.zero_()
        
        y = self.genometrans(emb)
        
        x = torch.cat([x, y], axis=-1)
        
        x = self.trans(x)
        x = self.lstm(x)[0]
        x = self.tail(x)
        
        if utils.half_length > self.halfseqlen:
            x = nn.functional.pad(x, (0, 0, utils.half_length-self.halfseqlen, utils.half_length-self.halfseqlen), value=0)
            
        if not self.oh:
            x = torch.sigmoid(x[:, utils.half_length, -1:])
            return x
        else:
            return x[:, utils.half_length]
    
class MultiSRAMP(nn.Module):
    def __init__(self, mode='full', halfseqlen=400, oh=True, sramp=None):
        super().__init__()
        if not sramp: self.sramp = SRAMP(mode=mode, halfseqlen=halfseqlen, oh=oh)
        else: self.sramp = sramp

    def forward(self, x, emb, *args): # Need improve
        # bs, t, l, e = emb.shape
        # x = x.reshape((bs*t, l))
        # emb = emb.reshape((bs*t, l, e))
        
        # x = self.sramp(x, emb, *args)
        # x = x.reshape((bs, t, 2))
        
        # x = nn.functional.softmax(x[:, :, 0], dim=-1) * x[:, :, -1]
        # x = x.sum(axis=-1).unsqueeze(-1)

        # return x

        
        res = []
        for i in range(utils.max_t):
            if emb[:, i].mean() == -1: break
            res += [self.sramp(x[:, i], emb[:, i], *args).unsqueeze(1)]
            
        x = torch.cat(res, axis=1)
        x = nn.functional.softmax(x[:, :, 0], dim=-1) * x[:, :, -1]
        x = x.sum(axis=-1).unsqueeze(-1)
        
        return x

    
class TriLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        loss = 0
        loss += nn.functional.binary_cross_entropy_with_logits(x[:, -1], y[:, -1])
        return loss

sw = 2
class DeepPromise(nn.Module):
    def __init__(self, mode='enac'):
        super().__init__()
        self.mode = mode
        self.oh = Onehot(6)
        self.uf = torch.nn.Unfold((sw, 5))
        self.emb = nn.Linear(5, 5)
        self.conv1 = nn.Sequential(
            self.gen_conv_block(5, 64),
            nn.Dropout(0.2),
            self.gen_conv_block(64, 64),
            nn.Dropout(0.2),
            self.gen_conv_block(64, 64),
            nn.Dropout(0.2),
            self.gen_conv_block(64, 64),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(3712, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )
        
    def gen_conv_block(self, i, o):
        return nn.Sequential(
            nn.Conv1d(i, o, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
    def forward(self, x, emb, *args):
        b, l = x.shape
        x = self.oh(x)[:, :, :-1]
        
        if self.mode == 'enac':
            x = x.unsqueeze(1)
            x = self.uf(x).reshape((b, -1, 5, l-sw+1))
            x = x.sum(axis=1) / sw
        elif self.mode == 'embedding':
            x = self.emb(x).transpose(1, 2)
        elif self.mode == 'onehot':
            x = x.transpose(1, 2)
        else:
            raise Exception('wrong mode')
        x = self.conv1(x)
        x = self.head(x)[:, 0]
        return x

class DeepPromiseEnsemble(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = models
        self.w = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
        
    def forward(self, *x):
        device = self.w.device
        x = torch.cat([model(*x).unsqueeze(-1).detach().to(device) for model in self.models], dim=-1).to(device)
        x = (self.w * x).sum(axis=-1)
        # x = torch.clamp(x, 0, 1)
        return x
        

class DPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x, y):
        return self.loss(x, y[:, -1])
    
    
# dp = DeepPromise(mode='embedding')
# dp(torch.randint(4, (1, 601,)), 1, 1)
# dp(torch.ones((1, 10,), dtype=torch.long), 1, 1)
