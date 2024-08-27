import bigram_lookup
import single_layer_nn
import mlp
import batch_norm

def main():
    words = open('Makemore/names.txt').read().splitlines()
    bigram_lookup_model = bigram_lookup.BigramLookupModel(27)
    bigram_lookup_model.learn(words)
    print(bigram_lookup_model.predict(10))

    single_layer_nn_model = single_layer_nn.SingleLayerNN(27)
    single_layer_nn_model.learn(words)
    print(single_layer_nn_model.predict(10))

    mlp_model = mlp.MLPModel(30)
    mlp_model.learn(words)
    print(mlp_model.predict(20))

    batch_norm_model = batch_norm.BatchNorm(30)
    batch_norm_model.learn(words)
    print(batch_norm_model.predict(20))

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()