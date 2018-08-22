import numpy as np


'''
A sample of saved data

5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
'''


# Load data from file, shuffle and split
def load_data(file_name, frac_as_test=0.15):
    with open(file_name, 'r') as reader:
        lines = reader.readlines()

        # Load x_data with integer values
        x_data = [[float(i) for i in line.split(',')[:-1]] for line in lines]
        # Load y_data with class values
        y_data = [line.split(',')[-1].rstrip() for line in lines]
        # y_data = [0 if element == 'Iris-setosa' else element for element in y_data]

        # Shuffle data
        idx = [i for i in range(len(y_data))]
        np.random.shuffle(idx)

        x_data = np.array([x_data[curr_idx] for curr_idx in idx])
        y_data = [y_data[curr_idx] for curr_idx in idx]

        # split data
        n_test = int(len(y_data) * frac_as_test)
        x_test = x_data[:n_test]
        y_test = y_data[:n_test]

        x_train = x_data[n_test:]
        y_train = y_data[n_test:]

    return x_train, y_train, x_test, y_test


def knn_classifier(x_tr, y_tr, x_te, y_te, knn=5):
    corrects = 0
    for i in range(len(y_te)):
        temp_x_tr = np.copy(x_tr)
        temp_y_tr = y_tr.copy()
        test_x = x_te[i: i+1]

        temp_x_tr -= test_x

        temp_x_tr = np.linalg.norm(temp_x_tr, axis=-1)

        temp_y_tr = [y for x, y in sorted(zip(temp_x_tr, temp_y_tr))]
        temp_x_tr = sorted(temp_x_tr)

        predicted = max(set(temp_y_tr[:knn]), key=temp_y_tr[:knn].count)

        print("Input test data :", test_x)
        print("Predicted O/P", predicted, "\tActual O/P", y_te[i], "\n")

        if predicted == y_te[i]:
            corrects += 1

    print("Accuracy :", corrects/float(len(y_te)) * 100, "%")


if __name__ == '__main__':
    file_name = 'iris.txt'
    # X values are numpy arrays whereas y is a list of strings
    x_tr, y_tr, x_te, y_te = load_data(file_name)

    knn_classifier(x_tr, y_tr, x_te, y_te, knn=3)
