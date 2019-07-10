from sakurai_nmf import benchmark_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = benchmark_model.load_hitachi_data(args.path, test_size=0.1)
    assert x_train.shape == (2991, 3)
    assert y_train.shape == (2991, 4)
