
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TasNet(nn.Module):
    def __init__(self, L, N, hidden_size, num_layers,EPS=1e-8,
            bidirectional=True, nspk=2,fc=0,e_type='conv'):
        super(TasNet, self).__init__()
        self.L, self.N, self.hidden_size, self.num_layers,self.bidirectional,self.nspk = L, N, hidden_size, num_layers,bidirectional, nspk
        self.EPS = EPS
        self.encoder = Encoder(L, N,EPS,e_type)
        self.separator = Separator(N, hidden_size, num_layers,
                                   bidirectional=bidirectional, nspk=nspk)
        self.decoder = Decoder(N, L)
        self.fc = fc
    def forward(self,mixture,mixture_lengths):
        mixture_e,norm_coef = self.encoder(mixture)
        pred_mask = self.separator(mixture_e,mixture_lengths)
        pred_source = self.decoder(mixture_e,pred_mask,norm_coef)

        return pred_source 
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['L'], package['N'],
                    package['hidden_size'], package['num_layers'],
                    bidirectional=package['bidirectional'],
                    nspk=package['nspk'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'L': model.L,
            'N': model.N,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'bidirectional': model.bidirectional,
            'nspk': model.nspk,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
class Encoder(nn.Module):
    def __init__(self,length,signal_size,EPS,e_type) -> None:
        super().__init__()
        self.e_type = e_type
        self.signal_size = signal_size
        if(e_type == 'fc'):
            self.fc1 = nn.Linear(length,signal_size,bias=False)
            self.fc2 = nn.Linear(length,signal_size,bias=False)
        elif (e_type=='conv'):
            self.conv1d_U = nn.Conv1d(length, signal_size, kernel_size=1, stride=1, bias=False)
            self.conv1d_V = nn.Conv1d(length, signal_size, kernel_size=1, stride=1, bias=False)
        else:
            print('ERROR')
            return 
        self.EPS = EPS
    def forward(self,mixture):
        B,K,L = mixture.size()
        norm_coef = torch.norm(mixture, p=2, dim=2, keepdim=True)  # B x K x 1
        norm_mixture = mixture/(norm_coef + self.EPS)
        # print(norm_mixture.shape)
        if self.e_type == 'fc':
            hidden = self.fc1(norm_mixture)
            hidden = F.relu(hidden)
            gate = self.fc2(norm_mixture)
            gate = torch.sigmoid(gate)
            y = hidden * gate
        else:
            norm_mixture = torch.unsqueeze(norm_mixture.view(-1, L), 2)  # B*K x L x 1
            hidden = F.relu(self.conv1d_U(norm_mixture))         # B*K x N x 1
            gate = torch.sigmoid(self.conv1d_V(norm_mixture))  # B*K x N x 1
            y = hidden * gate  # B*K x N x 1
            y = y.view(B, K, self.signal_size) # B x K x N
        return y,norm_coef
class Separator(nn.Module):
    def __init__(self, N, hidden_size, num_layers, bidirectional=True, nspk=2):
        super(Separator, self).__init__()
        # hyper-parameter
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        
        self.layer_norm = nn.LayerNorm(N)
        self.rnn = nn.LSTM(N, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=bool(bidirectional))
        
        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, nspk * N)

    def forward(self,mixture_e,mixture_lengths):
        """
        Args:
            mixture_e :after encoder,[B, K, N]
        Returns:
            est_mask: [B, K, nspk, N]
        """

        B,K,N = mixture_e.size()
        mixture_lengths = mixture_lengths.cpu()
        norm_mixture_e = self.layer_norm(mixture_e)
        packed_input = pack_padded_sequence(norm_mixture_e, mixture_lengths,
                                            batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=K)
        # print(output.shape)
        logits = self.fc1(output).view(B,K,self.nspk,N)
        est_mask = F.softmax(logits,dim=2)

        return est_mask
    
    
class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # hyper-parameter
        self.N, self.L = N, L

        self.fc1 = nn.Linear(N,L)
    def forward(self,mixture_e,est_mask,norm_coef):
        """
        Args:
            mixture_e: [B, K, N]
            est_mask: [B, K, nspk, N]
            norm_coef: [B, K, 1]
        Returns:
            est_source: [B, nspk, K, L]
        """
        source_e = torch.unsqueeze(mixture_e, 2) * est_mask  # B x K x nspk x N
        est_source = self.fc1(source_e)
        norm_coef = torch.unsqueeze(norm_coef, 2) #恢复原始大小
        est_source = est_source.permute((0, 2, 1, 3)).contiguous() # B x nspk x K x L
        return est_source
