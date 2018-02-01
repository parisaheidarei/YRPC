
1 data_root = 'C:/Users/parisa/Desktop/csYRPCai/project/train/'
2 
3 import numpy as np
4 import pandas as pd 
5 
6 train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
7 train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv', index8 _col='photo_id')
9 
10 train_df = pd.read_csv(data_root+"train_biz_HOGfeatures.csv")
 
11 y_train = train_df['label'].values
12 X_train = train_df['feature vector'].values
13  
14 def convert_label_to_array(str_label):
15     str_label = str_label[1:-1]
16     str_label = str_label.split(',')
17     return [int(x) for x in str_label if len(x)>0]
18 
19 def convert_feature_to_vector(str_feature):
20     str_feature = str_feature[1:-1]
21     str_feature = str_feature.split(',')
22     return [float(x) for x in str_feature]
23  
24 y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
25 X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
26 
27 print "X_train: ", X_train.shape
28 print "y_train: ", y_train.shape
29 #print "train_df: ", train_df[0:5]
30 
31 from sklearn import svm, datasets
32 from sklearn.cross_validation import train_test_split
33 from sklearn.preprocessing import label_binarize
34 from sklearn.multiclass import OneVsRestClassifier
35 from sklearn.preprocessing import MultiLabelBinarizer
36  
37 import time
38 t=time.time()
39 
40 mlb = MultiLabelBinarizer()
41 y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix
42 
43 random_state = np.random.RandomState(0)
44 X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(X_train, y_train, test_size=.2,random_state=random_state)
45 
46 #classifier set up
47 classifier1 = OneVsRestClassifier(svm.SVC(C=4, kernel='linear', probability=True))
48 classifier2 = OneVsRestClassifier(svm.SVC(C=2, kernel='linear', probability=True))
49 classifier3 = OneVsRestClassifier(svm.SVC(C=0.25, kernel='linear', probability=True))
50 k4 = 15
51 classifier5 = OneVsRestClassifier(svm.SVC(C=2, kernel='linear', probability=True))
52 classifier6 = OneVsRestClassifier(svm.SVC(C=0.25, kernel='linear', probability=True))
53 classifier7 = OneVsRestClassifier(svm.SVC(C=0.5, kernel='linear', probability=True))
54 k8 = 19
55 classifier9 = OneVsRestClassifier(svm.SVC(C=0.5, kernel='linear', probability=True))
56  
57 y_ppredict = np.zeros(y_ptest.shape)
58  
59 classifier1.fit(X_ptrain, y_ptrain[:, 0])
60 y_ppredict[:, 0] = classifier1.predict(X_ptest)
61 print "finish 1"
62 
63 classifier2.fit(X_ptrain, y_ptrain[:, 1])
64 y_ppredict[:, 1] = classifier2.predict(X_ptest)
65 print "finish 2"
66 
67 classifier3.fit(X_ptrain, y_ptrain[:, 2])
68 y_ppredict[:, 2] = classifier3.predict(X_ptest)
69 print "finish 3"
70
71 classifier5.fit(X_ptrain, y_ptrain[:, 4])
72 y_ppredict[:, 4] = classifier5.predict(X_ptest)
73 print "finish 5"
74 
75 classifier6.fit(X_ptrain, y_ptrain[:, 5])
76 y_ppredict[:, 5] = classifier6.predict(X_ptest)
77 print "finish 6"
78 
79 classifier7.fit(X_ptrain, y_ptrain[:, 6])
80 y_ppredict[:, 6] = classifier7.predict(X_ptest)
81 print "finish 7"
82  
83 classifier9.fit(X_ptrain, y_ptrain[:, 8])
84 y_ppredict[:, 8] = classifier9.predict(X_ptest)
85 print "finish 9"
86 for i, test_sample in enumerate(X_ptest):
87     #calculate distance
88     dist = [np.linalg.norm(test_sample - train_sample) for train_sample in X_ptrain]
89 
90     #sort
91     indices = np.argsort(dist)
92     
93     #predict 4 and 8
94     sum4 = 0
95     sum8 = 0
96     for j in xrange(k4):
97         sum4 = sum4 + y_ptrain[indices[j]][3]
98     for j in xrange(k8):
99         sum8 = sum8 + y_ptrain[indices[j]][7]
100  
101     y_ppredict[i][3] = sum4 > k4/2.0
102     y_ppredict[i][7] = sum8 > k8/2.0
103  
104 print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
105 
106 print "Samples of predicted labels (in binary matrix):\n", y_ppredict[0:3]
107 print "\nSamples of predicted labels:\n", mlb.inverse_transform(y_ppredict[0:3])
108  
109 statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
110 statistics.loc["biz count"] = np.append(np.sum(y_ppredict, axis=0), len(y_ppredict))
111 pd.options.display.float_format = '{:.0f}%'.format
112 statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_ppredict) 
113 statistics
114 
115 from sklearn.metrics import f1_score
116 
117 print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro') 
118 print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)
