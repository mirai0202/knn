import sys

import pandas as pd

from _classes import KNN, Metrics


def main(k: int, train_name: str, test_name: str):
    X_train = pd.read_csv(f'../datasets/{train_name}.data', header=None)
    y_train = X_train.pop(X_train.columns[-1]).rename('target')

    X_test = pd.read_csv(f'../datasets/{test_name}.data', header=None)
    y_test = X_test.pop(X_test.columns[-1]).rename('target')

    model = KNN(k)
    model.fit(X_train, y_train)
    print(f'Accuracy: {Metrics.accuracy(y_test, model.predict(X_test))}')
    user_interface(model)


def user_interface(model: KNN):
    message = (f'Write \'exit\' to quit the program or '
               f'your own record to predict in format:\n{"float " * len(model.X.columns)}\n')

    while (user_input := input(message)) != 'exit':
        try:
            values = [float(i) for i in user_input.split()]
            print(f'\033[95m{model.predict(pd.DataFrame(values).transpose())[0]}\033[0m')
        except ValueError as e:
            print(f'Incorrect record format ({e.args[0]})')


if __name__ == '__main__':
    print(sys.argv)
    '''
        format: 'k' 'name of train dataset' 'name of test dataset'
        example: 5 iris iris.test
    '''
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])

