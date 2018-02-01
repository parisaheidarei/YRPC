 
1 data_root = 'C:/Users/parisa/Desktop/csYRPCai/project/train/'
2 
3 import numpy as np
4 import pandas as pd 
5 
6 train_df = pd.read_csv(data_root+"train_biz_HOGfeatures.csv")
7 test_df = pd.read_csv(data_root+"test_biz_HOGfeatures.csv")
8 
9 def convert_label_to_array(str_label):
10     str_label = str_label[1:-1]
11     str_label = str_label.split(',')
12     return [int(x) for x in str_label if len(x)>0]
13  
14 def convert_feature_to_vector(str_feature):
15     str_feature = str_feature[1:-1]
16     str_feature = str_feature.split(',')
17     return [float(x) for x in str_feature]
18  
19 y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
20 X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
21 X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
22 
23 print "X_train: ", X_train.shape
24 print "y_train: ", y_train.shape
25 #print "train_df: ", train_df[0:5]
26  
27 from sklearn import svm, datasets
28 from sklearn.cross_validation import train_test_split
29 from sklearn.preprocessing import label_binarize
30 from sklearn.multiclass import OneVsRestClassifier
31 from sklearn.preprocessing import MultiLabelBinarizer
32  
33 import time
34 t=time.time()
35  
36 mlb = MultiLabelBinarizer()
37 y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix
38  
39 #classifier set up
40 classifier1 = OneVsRestClassifier(svm.SVC(C=4, kernel='linear', probability=True))
41 classifier2 = OneVsRestClassifier(svm.SVC(C=2, kernel='linear', probability=True))
42 classifier3 = OneVsRestClassifier(svm.SVC(C=0.25, kernel='linear', probability=True))
43 k4 = 15
44 classifier5 = OneVsRestClassifier(svm.SVC(C=2, kernel='linear', probability=True))
45 classifier6 = OneVsRestClassifier(svm.SVC(C=0.25, kernel='linear', probability=True))
46 classifier7 = OneVsRestClassifier(svm.SVC(C=0.5, kernel='linear', probability=True))
47 k8 = 19
48 classifier9 = OneVsRestClassifier(svm.SVC(C=0.5, kernel='linear', probability=True))
49 
50 y_predict = np.zeros((10000, 9))
51 
52 classifier1.fit(X_train, y_train[:, 0])
53 y_predict[:, 0] = classifier1.predict(X_test)
54 print "finish 1"
55 
56 classifier2.fit(X_train, y_train[:, 1])
57 y_predict[:, 1] = classifier2.predict(X_test)
58 print "finish 2"
59 
60 classifier3.fit(X_train, y_train[:, 2])
61 y_predict[:, 2] = classifier3.predict(X_test)
62 print "finish 3"
63  
64 classifier5.fit(X_train, y_train[:, 4])
65 y_predict[:, 4] = classifier5.predict(X_test)
66 print "finish 5"
67  
68 classifier6.fit(X_train, y_train[:, 5])
69 y_predict[:, 5] = classifier6.predict(X_test)
70 print "finish 6"
71  
72 classifier7.fit(X_train, y_train[:, 6])
73 y_predict[:, 6] = classifier7.predict(X_test)
74 print "finish 7"
75 
76 classifier9.fit(X_train, y_train[:, 8])
77 y_predict[:, 8] = classifier9.predict(X_test)
78 print "finish 9"
79 for i, test_sample in enumerate(X_test):
80     #calculate distance
81     dist = [np.linalg.norm(test_sample - train_sample) for train_sample in X_train]
82  
83     #sort
84     indices = np.argsort(dist)
85      
86     #predict 4 and 8
87     sum4 = 0
88     sum8 = 0
89     for j in xrange(k4):
90         sum4 = sum4 + y_train[indices[j]][3]
91     for j in xrange(k8):
92         sum8 = sum8 + y_train[indices[j]][7]
93  
94     y_predict[i][3] = sum4 > k4/2.0
95     y_predict[i][7] = sum8 > k8/2.0
96  
97 y_predict_label = mlb.inverse_transform(y_predict)
98 print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
99  
100 print "Samples of predicted labels (in binary matrix):\n", y_predict[0:3]
101 print "\nSamples of predicted labels:\n", mlb.inverse_transform(y_predict[0:3])
102  
103 df = pd.DataFrame(columns=['business_id','labels'])
104  
105 for i in range(len(test_df)):
106     biz = test_df.loc[i]['business']
107     label = y_predict_label[i]
108    label = str(label)[1:-1].replace(",", " ")
109     df.loc[i] = [str(biz), label]
110 
111 with open(data_root+"submission_HOG.csv",'w') as f:
112     df.to_csv(f, index=False)  
