import torch
import torch.nn.functional as F
import random

class Linear:
    def __init__(self, fan_in, fan_out, include_bias=True, generator = torch.Generator().manual_seed(2147483647)):
        self.W = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.b = torch.zeros(fan_out) if include_bias else None
    
    def __call__(self, input):
        ret = input @ self.W 
        if self.b is not None:
            ret += self.b
        return ret

    def parameters(self):
        return [self.W] + [] if self.b is None else [self.b]
    
class TanH:
    def __call__(self, input):
        return torch.tanh(input)
    
    def parameters(self):
        return []
    
class BatchNorm1D:
    def __init__(self, size):
        self.bngain = torch.ones((1, size))
        self.bnbias = torch.zeros((1, size))
        self.bnmean_running = torch.ones((1, size)) 
        self.bnvar_running = torch.ones((1, size))
        self.eps = .00001

    def parameters(self):
        return [self.bngain, self.bnbias]
    
    def __call__(self, input):
        # We are using "is_grad_enabled" as short hand for training
        if torch.is_grad_enabled():
            mean = input.mean(0, keepdim = True)
            xvar = input.var(0, keepdim = True)
        else:
            mean = self.bnmean_running
            xvar = self.bnvar_running
        
        xhat = (input - mean)/torch.sqrt(xvar + self.eps)
        result = self.bngain * xhat + self.bnbias

        if torch.is_grad_enabled():
            with torch.no_grad():
                self.bnmean_running = self.bnmean_running * .999 + self.bnmean_running * .001
                self.bnvar_running = self.bnvar_running * .999 + self.bnvar_running * .001

        return result


    