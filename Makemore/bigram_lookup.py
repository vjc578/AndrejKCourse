import torch

class BigramLookupModel:
    def __init__(self, seed):
        self.g = torch.Generator()
        if seed is not None:
            self.g = self.g.manual_seed(seed)

    def learn(self, words):
        N = torch.zeros((27,27), dtype=torch.int32)
        chars = sorted(list(set(''.join(words))))
        self.stoi = {s:i+1 for i,s in enumerate(chars)}
        self.stoi['.'] = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip (chs, chs[1:]):
                ix = self.stoi[ch1]
                iy = self.stoi[ch2]
                N[ix][iy] += 1
        self.itos = {i:s for s,i in self.stoi.items()}
        self.P = N / N.sum(1, keepdim=True).float()

    def predict(self, num_predictions):
        result = []
        for i in range(num_predictions):
            ix = 0
            out = []
            while True:
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=self.g).item()
                out.append(self.itos[ix])
                if (ix == 0):
                    break
            result.append(''.join(out))
        return result
