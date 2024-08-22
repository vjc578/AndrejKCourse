import random
from value import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #w*x + b
        assert len(x) == len(self.w)
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin]+ nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def eval(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train_step(self, input_batch, input_batch_target):
        pred = [self.eval(input_line) for input_line in input_batch]
        loss = sum((prediction - target)**2 for prediction, target in zip(pred, input_batch_target))
        
        for param in self.parameters():
            param.grad = 0
            
        loss.backprop()
        
        for param in self.parameters():
            param.data += -.01 * param.grad

        return (pred,loss)

    def train_steps(self, count, input_batch, input_batch_targets):
        for _ in range(count):
            self.train_step(input_batch, input_batch_targets)