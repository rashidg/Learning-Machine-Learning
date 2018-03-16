import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def do_logistic_regression(train_data, train_labels, test_data, test_labels):
    print("=== | 75/25 split Logistic Regression |")
    from scipy.sparse import vstack
    all_data = vstack((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))
    splt = int(all_data.shape[0] * 0.75)
    train_data = all_data[:splt]
    train_labels = all_labels[:splt]
    test_data = all_data[splt:]
    test_labels = all_labels[splt:]

    model = LogisticRegression(tol=0.01, C=2.0, max_iter=500, solver='sag')
    model.fit(train_data, train_labels)

    score = model.score(train_data, train_labels)
    print("logistic train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("logistic test score: %.4f" % score)

def do_NB_multinomial(train_data, train_labels, test_data, test_labels):
    print("=== | 75/25 split | Naive Bayes Multinomial |")
    from scipy.sparse import vstack
    all_data = vstack((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))
    splt = int(all_data.shape[0] * 0.75)
    train_data = all_data[:splt]
    train_labels = all_labels[:splt]
    test_data = all_data[splt:]
    test_labels = all_labels[splt:]

    model = sklearn.naive_bayes.MultinomialNB(alpha=0.01)
    model.fit(train_data.toarray(), train_labels)

    score = model.score(train_data.toarray(), train_labels)
    print("Naive Bayes train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("Naive Bayes train score: %.4f" % score)

def do_knn(train_data, train_labels, test_data, test_labels):
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')

    model.fit(train_data, train_labels)

    score = model.score(train_data, train_labels)
    print("KNN train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("KNN test score: %.4f" % score)

def do_neural_net(train_data, train_labels, test_data, test_labels, feature_names):
    model = MLPClassifier(hidden_layer_sizes=(50, ), activation='logistic', solver='adam', learning_rate='constant',
                          alpha=0.0001,
                          max_iter=100,
                          tol=0.0001,
                          verbose=True,
                          momentum=0.2,
                          nesterovs_momentum=False,
                          early_stopping=True,
                          validation_fraction=0.1)

    print("=== | 75/25 split | Neural Nets |")
    from scipy.sparse import vstack
    all_data = vstack((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))

    splt = int(all_data.shape[0] * 0.75)
    train_data = all_data[:splt]
    train_labels = all_labels[:splt]
    test_data = all_data[splt:]
    test_labels = all_labels[splt:]

    model.fit(train_data, train_labels)
    score = model.score(train_data, train_labels)
    print("Neural Nets train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("Neural Nets test score: %.4f" % score)

    feature_count = len(feature_names)
    confusion_mat = np.zeros((feature_count, feature_count))
    pred = model.predict(test_data)
    for i in range(len(pred)):
        real_label = test_labels[i]
        pred_label = pred[i]
        confusion_mat[pred_label][real_label] += 1

    np.set_printoptions(threshold=10000)
    print("Confusion Matrix:")
    print(confusion_mat)


def do_svm(train_data, train_labels, test_data, test_labels):
    model = SVC(C=1.0, kernel='poly', degree=2, gamma='auto', tol=0.01, max_iter=800, decision_function_shape='ovr')
    model.fit(train_data, train_labels)

    score = model.score(train_data, train_labels)
    print("SVM train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("SVM test score: %.4f" % score)

def do_gaussian_nb(train_data, train_labels, test_data, test_labels):
    model = GaussianNB()
    print("=== | 75/25 split | Naive Bayes Multinomial |")
    from scipy.sparse import vstack
    all_data = vstack((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))

    splt = int(all_data.shape[0] * 0.75)
    train_data = all_data[:splt]
    train_labels = all_labels[:splt]
    test_data = all_data[splt:]
    test_labels = all_labels[splt:]

    model.fit(train_data.toarray(), train_labels)

    score = model.score(train_data.toarray(), train_labels)
    print("Gaussian Naive Bayes train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("Gaussian Naive Bayes test score: %.4f" % score)

def do_DT_adaboost(train_data, train_labels, test_data, test_labels):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                               n_estimators=100,
                               learning_rate=1)

    model.fit(train_data, train_labels)
    score = model.score(train_data, train_labels)
    print("Decision Tree train score: %.4f" % score)

    score = model.score(test_data, test_labels)
    print("Decision Tree test score: %.4f" % score)

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = tf_idf_features(train_data, test_data)

    do_logistic_regression(train_bow, train_data.target, test_bow, test_data.target)
    do_NB_multinomial(train_bow, train_data.target, test_bow, test_data.target)
    do_neural_net(train_bow, train_data.target, test_bow, test_data.target, feature_names)


    # DT
    # do_DT_adaboost(train_bow, train_data.target, test_bow, test_data.target)
    # KNN
    #do_knn(train_bow, train_data.target, test_bow, test_data.target)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
