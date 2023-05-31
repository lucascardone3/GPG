import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  print(array)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Circulo")
  elif answer == 1:
    print("pred: Cuadrado")
  elif answer == 2:
    print("pred: Triangulo")

  return answer

predict("descarga.jpg")
