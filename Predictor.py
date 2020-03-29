from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('covid19.model')

test_image = image.load_img('test.jpeg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)
print(max(result))
