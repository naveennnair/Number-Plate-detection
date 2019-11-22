import os
import numpy as np
#from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#os.chdir()

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y'
        ]

width = 30
height = 30

def resize_img(training_directory):
    for each_letter in letters:
        for each in range(16):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            im = Image.open(image_path)
            im = im.resize((width, height), Image.NEAREST)
            im.save(image_path)             
            
def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)

# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')
training_dataset_dir = './Training_images'
resize_img(training_dataset_dir)
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')

# Selecting model

#svc_model = SVC(kernel='linear', probability=True)
rand_model = RandomForestClassifier(n_estimators= 1000, max_depth=7)
#xgb_model  = XGBClassifier(n_estimators = 1000)
#cross_validation(xgb_model, 4, image_data, target_data)

print('training model')

# let's train the model with all the input data
#svc_model.fit(image_data, target_data)
rand_model.fit(image_data, target_data)


import pickle
print("model trained.saving model..")
filename = './finalized_model.sav'
pickle.dump(rand_model, open(filename, 'wb'))
print("model saved")



