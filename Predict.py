from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

test_model = load_model('first_model.h5')
img = load_img('2.jpg',False,target_size=(150,150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)

if preds[0][0] == 0:
    print("Covid Lungs")

elif preds[0][0] == 1:
    print("Normal Lungs")

# 1 is normal lungs
# 0 is covid lungs