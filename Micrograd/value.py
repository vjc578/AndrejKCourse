import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truedev__(self, other):
        return self * other**-1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), 'sub')
        return out

    def __rsub__(self, other):
        return -1 * (self - other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), 'pow')
        # Gross
        out.power = other
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out

    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, (self, ), 'exp')
        return out

    def backprop(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            if node._op == '+':
                for child in node._prev:
                    child.grad += node.grad

            # Could do self excluding product if there is more.
            elif node._op == '*':
                node._prev[0].grad += node._prev[1].data * node.grad
                node._prev[1].grad += node._prev[0].data * node.grad

            elif node._op == 'tanh':
                node._prev[0].grad += node.grad * (1 - node.data**2)   

            elif node._op == 'exp':
                node._prev[0].grad += node.grad * node.data

            elif node._op == 'pow':
                prev = node._prev[0]
                prev.grad += node.grad * (node.power * prev.data**(node.power - 1))

            elif node._op == 'sub':
                node._prev[0].grad = node.grad
                node._prev[1].grad = -1 * node.grad

class ValueGrapher:
    def trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes,edges

    def draw_dot(root):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        
        nodes, edges = ValueGrapher.trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(name = uid, label = "{%s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad ), shape='record')
            if (n._op):
                dot.node(name = uid + n._op, label = n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        dot.render('doctest-output/graph.gv', view=True)              