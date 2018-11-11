#Author: Suleka Helmini (https://github.com/suleka96)

#this model was designed to test using the Enron datase (pre-processed version)
#it can be obtained through the blow link.
#https://www.kaggle.com/crawford/20-newsgroups
#If you are running this in your local machine, download the dataset and put all the emails into one folder called 'emails' which is inside a folder called 'enron'.
#or change the 'direc' to where your emails are

import os
from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def make_dict():
    direc = "enron/email2/"
    files = os.listdir(direc)
    emails = [direc+email for email in files]

    words = []

    for email in emails:
        f = open(email,encoding="utf8", errors='ignore')
        blob = f.read()
        words += blob.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]

    return dictionary.most_common(3000)

def make_dataset(dictionary):
    direc = "enron/email2/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    print(len(emails))

    feature_set = []
    labels = []

    for email in emails:
        data = []
        f = open(email, encoding="utf8", errors='ignore')
        words = f.read().split(' ')

        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)
        if "ham" in email:
            labels.append(0)
        else:
            labels.append(1)

    print(feature_set[0])
    return feature_set, labels

def getScores(y, pred, name):
    print("--------------------- ",name," --------------------")
    print("Accuracy score")
    print(accuracy_score(y, pred))
    print("F1 score")
    print(f1_score(y, pred, average='macro'))
    print("Recall")
    print(recall_score(y, pred, average='macro'))
    print("Precision")
    print(precision_score(y, pred, average='macro'))

def rfc(train_x,train_y,val_x, val_y,test_x,test_y):

    print("---------------Random Forest Classifier---------")

    rcf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=None)
    rcf.fit(train_x, train_y)
    validation_pred = rcf.predict(val_x)
    test_pred = rcf.predict(test_x)

    getScores(val_y,validation_pred, "Validation Scores")
    getScores(test_y, test_pred,"Test Scores")

def svc(train_x,train_y,val_x, val_y,test_x,test_y):

    print("----------------------SVC------------------------------")

    svctest2 = SVC(kernel='linear', random_state=42, C=1)
    svctest2.fit(train_x, train_y)
    validation_pred = svctest2.predict(val_x)
    test_pred = svctest2.predict(test_x)

    getScores(val_y,validation_pred, "Validation Scores")
    getScores(test_y, test_pred,"Test Scores")

def logisticRegression(train_x,train_y,test_x,test_y):

    print("----------------------Logistic Regression------------------------------")

    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)

    getScores(test_y,prediction,"Test Scores")

if __name__ == '__main__':
    dictionary = make_dict()
    features, labels = make_dataset(dictionary)

    split_frac1 = 0.8

    idx1 = int(len(features) * split_frac1)
    train_x, val_x = features[:idx1], features[idx1:]
    train_y, val_y = labels[:idx1], labels[idx1:]

    split_frac2 = 0.5

    idx2 = int(len(val_x) * split_frac2)
    val_x, test_x = val_x[:idx2], val_x[idx2:]
    val_y, test_y = val_y[:idx2], val_y[idx2:]

    logisticRegression(train_x,train_y,test_x,test_y)
    svc(train_x,train_y,val_x, val_y,test_x,test_y)
    rfc(train_x, train_y, val_x, val_y, test_x, test_y)




