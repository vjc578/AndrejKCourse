import value
from nn import MLP

def main():
    mlp = MLP(3, [4, 3, 1])
    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    for i in range(100):
        x = mlp.train_step(xs, ys)
    print(x)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()