import torch
import torch.nn.functional as F

class SingleLayerNN:
    def __init__(self, seed):
        self.g = torch.Generator()
        if seed is not None:
            self.g = self.g.manual_seed(seed)

    def learn(self, words):
        # Create the training set of all the bigrams
        xs, ys = [], []
        chars = sorted(list(set(''.join(words))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip (chs, chs[1:]):
                ix = stoi[ch1]
                iy = stoi[ch2]
                xs.append(ix)
                ys.append(iy)
        self.itos = {i:s for s,i in stoi.items()}

        # Lowercase t keeps the type, uppercase T (.Tensor) converts to float
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        self.W = torch.randn((27,27), generator=self.g, requires_grad=True)
        xenc = F.one_hot(xs, num_classes=27).float()

        for i in range(100):
            ### Forward pass ####
            
            #(N x 27) @ (27 X 27) = (N X 27)
            logits = xenc @ self.W
            
            # For each example probability of next character.
            # Next two steps are softmax (makes NN output probabilities, outputs of all layers sums to 1)
            # This at the end is essentially the same as our original probabilitiy matrix.
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdims=True)
            # Tries to push W towards zero. Avoid overfitting.
            reg = 0.01*(self.W**2).mean()
            loss = -prob[torch.arange(xs.nelement()), ys].log().mean() + reg

            ### Backwards ####
            
            self.W.grad = None # set to zero the gradient
            loss.backward()
            self.W.data += -50 * self.W.grad

    def predict(self, num_predictions):
        result = []
        for i in range(num_predictions):
            ix = 0
            out = []
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ self.W
                counts = logits.exp()
                prob = counts / counts.sum(1, keepdims=True)

                ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=self.g).item()
                out.append(self.itos[ix])
                if (ix == 0):
                    break
            result.append(''.join(out))
        return result
