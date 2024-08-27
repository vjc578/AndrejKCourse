import torch
import torch.nn.functional as F
import random
import nn

class BatchNorm:
    def __init__(self, seed):
        self.g = torch.Generator()
        if seed is not None:
            self.g = self.g.manual_seed(seed)

    def _build_dataset(self, words):
        self.blocksize = 3
        words = open('Makemore/names.txt').read().splitlines()
        chars = sorted(list(set(''.join(words))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        self.itos = {i:s for s,i in stoi.items()}
        self.vocab_size = len(self.itos)

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
        [self.X, self.Y] = build_dataset(words[:n1])
        [self.XV, self.YV] = build_dataset(words[n1:n2])
        [self.XT, self.YT] = build_dataset(words[n2:])

    def learn(self, words):
        self._build_dataset(words)

        self.embedding_dimensions = 10
        self.C = torch.randn((self.vocab_size, self.embedding_dimensions), generator=self.g, requires_grad=True)
        self.embedding_sum_size = self.embedding_dimensions * self.blocksize
        self.hidden_layer_size = 200

        self.layers = [
            nn.Linear(self.embedding_sum_size, self.hidden_layer_size),
            nn.BatchNorm1D(self.hidden_layer_size),
            nn.TanH(),
            nn.Linear(self.hidden_layer_size, self.vocab_size)
        ]

        parameters = [p for layer in self.layers for p in layer.parameters()]
        for p in parameters:
            p.requires_grad = True

        def forward(emb, target):
            x = emb.view(-1, self.embedding_sum_size)
            for layer in self.layers:
                x = layer(x)

            # Cross entropy loss
            anll = F.cross_entropy(x, target)

            return anll
    
        for _ in range(10000):
            # Mini batch size 64
            ix = torch.randint(0, self.X.shape[0], (64, ))

            # Embedded values, shape (64, embedding_size)
            emb = self.C[self.X[ix]]
            
            # Cross entropy loss
            anll = forward(emb, self.Y[ix])

            # Backwards Pass
            for p in parameters:
                p.grad = None
            anll.backward()
            
            # Stochastic gradient descent
            learning_rate = 0.05
            for p in parameters:
                p.data += -learning_rate * p.grad
                    
        # Evaluate current loss.
        with torch.no_grad():
            print(f'Train: {forward(self.C[self.X], self.Y)} Validation: {forward(self.C[self.XV], self.YV)} Test: {forward(self.C[self.XT], self.YT)}')

    @torch.no_grad()
    def predict(self, num_predictions):
        ret_val = []
        for _ in range(num_predictions):
            out = []
            context = [0] * self.blocksize
            while True:
                emb = self.C[torch.tensor([context])]
                x = emb.view(-1, self.embedding_sum_size)
                for layer in self.layers:
                    x = layer(x)
                probs = F.softmax(x, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            ret_val.append(''.join(self.itos[i] for i in out))
        return ret_val            