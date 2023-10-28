import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

model = load_model("resnet60.h5")
classes=['Hispa', 'Healthy', 'BrownSpot', 'LeafBlast']

def check_paddy_health(img_path):
    image = load_img(img_path, grayscale=False, color_mode='rgb', target_size=(80,80))
    image = img_to_array(image)
    image /= 255
    prediction = model.predict([image])
    return classes[prediction[0]]