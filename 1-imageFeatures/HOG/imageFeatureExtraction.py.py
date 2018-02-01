Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:01:18) [MSC v.1900 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> 
1 data_root = 'C:/Users/parisa/Desktop/csYRPCai/project/train/'
2
3 import os
4 import time
5 import numpy as np
6 import matplotlib.pyplot as plt
7 from skimage.feature import hog
8 from skimage.transform import resize
9 from skimage import io, data, color, exposure
10 import h5py
11 import pandas as pd 
12 
13 def extract_hog_features(image_path):
14     image = io.imread(image_path)
15     image = color.rgb2gray(image)
16     image_resized = resize(image, (256, 256))
17     return hog(image_resized, orientations=8,
18         pixels_per_cell=(16, 16), cells_per_block=(1, 1))
19  
20 f = h5py.File(data_root+'train_image_HOGfeatures.h5','w')
21 filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
22 feature = f.create_dataset('feature',(0,2048), maxshape = (None,2048))
23 f.close()
24 
25 train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
26 train_folder = data_root+'train_photos/'
27 train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photo28 s['photo_id']]  # get full filename
29  
30 num_train = len(train_images)
31 print "Number of training images: ", num_train
32 
33 tic = time.time()
34 
35 # Training Images
36 for i in range(0, num_train):
37     feature = extract_hog_features(train_images[i])
38     num_done = i+1
39     f= h5py.File(data_root+'train_image_HOGfeatures.h5','r+')
40     f['photo_id'].resize((num_done,))
41     f['photo_id'][i] = train_images[i]
42     f['feature'].resize((num_done,feature.shape[0]))
43     f['feature'][i, :] = feature
44     f.close()
45     if num_done%1000==0 or num_done==num_train:
46         print "Train images processed: ", num_done
47 
48 toc = time.time()
49 print '\nFeatures extracted in %fs' % (toc - tic)
