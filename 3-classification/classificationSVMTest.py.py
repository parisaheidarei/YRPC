 
data_root = 'C:/Users/parisa/Desktop/csYRPCai/project/train/'
 
import numpy as np
import pandas as pd
import time
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer
 
train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv', index_col='photo_id')
train_df = pd.read_csv(data_root+"train_biz_HOGfeatures.csv")
 
test_photos = pd.read_csv(data_root+'test_photo_to_biz.csv')
test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv', index_col='photo_id')
test_df = pd.read_csv(data_root+"test_biz_HOGfeatures.csv")
 
y_train = train_df['label'].values
X_train = train_df['feature vector'].values
X_test = test_df['feature vector'].values
 
def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]
 
def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]
 
y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
 
print "X_train: ", X_train.shape
print "X_test: ", X_test.shape
print "y_train: ", y_train.shape
 
t=time.time()
 
mlb = MultiLabelBinarizer()
y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix
 
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
#classifier = OneVsRestClassifier(svm.SVC(C=1.0, gamma='auto', kernel='rbf', probability=True))
 
classifier.fit(X_train, y_train)
 
y_predict = classifier.predict(X_test)
 
print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
 
print "Samples of predicted labels (in binary matrix):\n", y_predict[0:3]
print "\nSamples of predicted labels:\n", mlb.inverse_transform(y_predict[0:3])
 
df = pd.DataFrame(columns=['business_id','labels'])
 
for i in range(len(test_df)):
    biz = test_df.loc[i]['business']
    label = y_predict[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]
 
with open(data_root+"submission_HOG.csv",'w') as f:
    df.to_csv(f, index=False)
