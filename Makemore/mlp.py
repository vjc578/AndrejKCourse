import torch
import torch.nn.functional as F
import random

class MLPModel:
    def __init__(self, seed):
        self.g = torch.Generator()
        if seed is not None:
            self.g = self.g.manual_seed(seed)

    def learn(self, words):
        self.blocksize = 3

        words = open('Makemore/names.txt').read().splitlines()
        chars = sorted(list(set(''.join(words))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        self.itos = {i:s for s,i in stoi.items()}

        def build_dataset(words):
            X = []
            Y = []
            for word in words:
                context = [0] * self.blocksize
                for ch in word + '.':
                    ix = stoi[ch]
                    X.append(context)
                    Y.append(ix)
                    context = context[1:] + [ix]
            
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            return X,Y
        
        random.seed(42)
        random.shuffle(words)
        n1 = int(0.8*len(words))
        n2 = int(0.9*len(words))
        [X, Y] = build_dataset(words[:n1])
        [XV, YV] = build_dataset(words[n1:n2])
        [XT, YT] = build_dataset(words[n2:])

        self.C = torch.randn((27, 10), generator=self.g, requires_grad=True)
        self.embedding_sum_size = self.C.shape[1] * self.blocksize

        self.W1 = torch.randn((self.embedding_sum_size, 200), generator=self.g, requires_grad=True)
        self.b1 = torch.randn(self.W1.shape[1], generator=self.g, requires_grad=True)
        self.W2 = torch.randn((self.W1.shape[1], 27), generator=self.g, requires_grad=True)
        self.b2 = torch.randn(self.W2.shape[1], generator=self.g, requires_grad=True)
        parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        def forward(emb, target):
            h = torch.tanh(emb.view(-1, self.embedding_sum_size) @ self.W1 + self.b1)
            # Output layer
            logits = h @ self.W2 + self.b2
            
            # Cross entropy loss
            anll = F.cross_entropy(logits, target)

            return anll

        for _ in range(20000):
            ix = torch.randint(0, X.shape[0], (64, ))
            # Forward pass
            # Embedded values
            emb = self.C[X[ix]]
            
            # Cross entropy loss
            anll = forward(emb, Y[ix])

            for p in parameters:
                p.grad = None
                
            anll.backward()
            
            learning_rate = 0.1
            for p in parameters:
                p.data += -learning_rate * p.grad
                
        print(f'Train: {forward(self.C[X], Y)} Validation: {forward(self.C[XV], YV)} Test: {forward(self.C[XT], YT)}')
        
    def predict(self, num_predictions):
        ret_val = []
        for _ in range(num_predictions):
            out = []
            context = [0] * self.blocksize
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            ret_val.append(''.join(self.itos[i] for i in out))
        return ret_val
