import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.callbacks import EarlyStopping
from PIL import Image
import pandas as pd
import sys


def gray_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('L', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img, 'L')
  newImage = quantizetopalette(oldImage,palImage, mode="L")
  res_image = np.asarray(newImage)
  return res_image

#Función para dada una paleta solo tomar los colores de esa paleta en la imagen
def quantizetopalette(silf, palette, dither=False, mode="P"):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  palette.load()
  im = silf.im.convert(mode, 0, palette.im)
  # the 0 above means turn OFF dithering making solid colors
  return silf._new(im)

#Toma todos los colores existentes en la imagen
def get_colors(image):
  aux = []
  band = True
  for i in image:
    for j in i:

      for k in aux:
        if j.tolist() == k:
          band = False
          break
      if band:
        aux.append(j.tolist())
      band = True
  return np.array(aux)

def get_colors_optimized(image):
    # Aplanar la imagen a una lista de píxeles (forma: número de píxeles, canales)
    pixels = image.reshape(-1, image.shape[-1])
    
    # Utilizar np.unique para encontrar filas únicas (colores únicos) en los píxeles aplanados
    # axis=0 opera a lo largo del eje de las filas para encontrar filas únicas
    # return_counts=False para no retornar los conteos de cada color único
    unique_colors = np.unique(pixels, axis=0, return_counts=False)
    
    return unique_colors

def recolor_greys_image(data, palette):
    rows, cols = len(data), len(data[0])
    aux = np.zeros((rows, cols), dtype=np.uint64)
    for i in range(rows):
        for j in range(cols):
            aux[i,j] = min(palette, key= lambda x:abs(x-data[i,j]))
    return aux

def recolor_greys_image_optimized(data, palette):
    # Asegurarse de que la paleta y los datos estén en el mismo tipo de datos y rango
    palette = np.array(palette, dtype='float32')
    
    data = data.astype('float32')
    
    # Expandir las dimensiones de los datos y la paleta para la transmisión (broadcasting)
    data_expanded = data[:, :, np.newaxis]  # Forma ahora es (rows, cols, 1)
    palette_expanded = palette[np.newaxis, np.newaxis, :]  # Forma ahora es (1, 1, num_colors)
    
    # Calcular la diferencia absoluta entre cada píxel y cada color de la paleta
    abs_diff = np.abs(data_expanded - palette_expanded)
    
    # Encontrar el índice del color más cercano en la paleta para cada píxel
    indices_of_nearest = np.argmin(abs_diff, axis=2)
    
    # Mapear los índices a los valores de la paleta para obtener la imagen recoloreada
    recolored_image = palette[indices_of_nearest]
    
    return recolored_image

def agroup_window(data, window):
    new_data = [data[i:window+i] for i in range(len(data)-window+1)]
    return np.array(new_data)

def balance_img_categories(img, palette, balancer):
  #palette = np.sort(palette)
  rows = len(img)
  cols = len(img[0])
  print("rows: ", rows, "cols: ", cols)
  for i in range(rows):
    for j in range(cols):
      pos = np.where(palette == img[i,j])[0][0]
      print("pos: ", pos)
      img[i,j] = balancer[pos]
  return img

def gray_quantized_optimized(img, palette):
    # Ejemplo de uso
    # img es tu imagen en escala de grises como un array de NumPy
    # palette es tu paleta deseada como un array de NumPy con valores de escala de grises
    # res_image = gray_quantized_optimized(img, palette)
    # Asegurar que img es un array de NumPy
    img = np.array(img, dtype=np.uint8)
    
    # Crear una imagen PIL directamente desde el array de NumPy
    oldImage = Image.fromarray(img, 'L')
    
    # Convertir la imagen a modo 'P' utilizando la paleta proporcionada
    # Nota: La paleta debe ser ajustada al formato esperado por PIL si es necesario.
    newImage = oldImage.quantize(palette=Image.fromarray(palette, 'P'))
    
    # Convertir la imagen cuantizada de vuelta a un array de NumPy
    res_image = np.asarray(newImage)
    
    return res_image

#Crea cubos con su propia información de tamaño h
def get_cubes(data, h):
    new_data = []
    for i in range(0, len(data)-h):
        new_data.append(data[i:i+h])
    new_data = np.array(new_data)
    print(new_data.shape)
    return new_data


linkDeGuardado = "Resultados/ResultadoCompletoOldCategories/"

#leer una entrada de usuario por consola para variable de carpeta
carpeta = input("Ingrese el nombre de la carpeta: ")
print(carpeta)

bach_size=2
learning_rate = 0.005
epochs=300
patience = 10
window = 10

#crear carpeta si no existe
if not os.path.exists(linkDeGuardado+carpeta):
  os.makedirs(linkDeGuardado+carpeta)
else:
   print("La carpeta ya existe")

linkDeGuardado = linkDeGuardado+carpeta


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
   

    categories = np.array([0, 35, 70, 119, 177, 220, 255])
    categoriesBalanced = np.array([18, 54, 90, 126, 162, 198, 234])
    channels = 1
    
    rows = 120
    cols = 360

#carga de la data categorica
    #x = np.load("/media/mccdual2080/Almacenamiengto/SahirProjects/SahirReyes/dataSetAutoencoder/DatasetAutoencoder/DataSetLatentSpace/Npy/Balanced/V1/Dataset120x360GreysNewCategories.npy")
    x = np.load("/media/mccdual2080/Almacenamiengto/SahirProjects/SahirReyes/dataSetAutoencoder/DatasetAutoencoder/DataSetLatentSpace/Npy/Dataset120x360Greys.npy")
    #x = x/255
    print (x.shape)
    print ("type of x: ", x.dtype)
    x_train = x[:int(len(x)*.7)]
    x_test = x[int(len(x)*.7):]
    x_validation = x_train[int(len(x_train)*.8):]
    x_train = x_train[:int(len(x_train)*.8)]
    #x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
    #x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
    #x_test = x_test.reshape(len(x_test), window, rows, cols, channels)

    print("Forma de datos de entrenamiento: {}".format(x_train.shape))
    print("Forma de datos de validación: {}".format(x_validation.shape))
    print("Forma de datos de pruebas: {}".format(x_test.shape))

    print("colores de la imagen de entrenamiento: ", get_colors(x_train[150]))
    print("colores de la imagen de validación: ", get_colors(x_validation[100]))
    print("colores de la imagen de pruebas: ", get_colors(x_test[100]))

    input_shape = (120, 360, 1)

    encoder_input = keras.Input(shape=input_shape)
    print ("entrada encoder",encoder_input.shape)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    encoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    print("salida encoder",encoder_output.shape)

    # Definir la arquitectura del decoder
    decoder_input = keras.Input(shape=encoder_output.shape[1:])
    print ("entrada decoder",decoder_input.shape)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(decoder_input)
    x = layers.UpSampling2D((2, 2))(x)
    #x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)  # Output 

    strategy = tf.distribute.MirroredStrategy()
    # Crear el encoder y el decoder como modelos de Keras
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Crear el autoencoder concatenando el encoder y el decoder
    

    autoencoder_input = keras.Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    optim=Adam(learning_rate=learning_rate)#0.0005
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")
    autoencoder.compile(optimizer=optim, loss="binary_crossentropy")

    
    early_stopping = EarlyStopping(monitor='val_loss',  # Métrica a monitorear
                                   patience=patience,         # Número de épocas sin mejora después de las cuales se detendrá el entrenamiento
                                   restore_best_weights=True)  # Restaurar los mejores pesos.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience) #monitor='val_loss', factor=0.2, patience=5, min_lr=0.001

    
    # guardar en un archivo txt el historial de entrenamiento
    history = autoencoder.fit(x=x_train, y=x_train, epochs=epochs, batch_size=bach_size, validation_data=(x_validation, x_validation), callbacks=[early_stopping, reduce_lr])

    with open(os.path.join(linkDeGuardado, "training_history.txt"), "w") as f:
      f.write("Loss\n")
      f.write(str(history.history['loss']) + '\n')
      f.write("Validation Loss\n")
      f.write(str(history.history['val_loss']) + '\n')
      f.write("Learning Rate\n")
      f.write(str(history.history['lr']) + '\n')
      f.write("Epochs\n")
      f.write(str(history.epoch) + '\n')

    #autoencoder.fit(x= x_train,y= x_train,epochs=epochs, batch_size=bach_size,validation_data=(x_validation, x_validation),callbacks=[early_stopping,reduce_lr])


    # guardar en un archivo txt el resumen del modelo autoencoder y los datos de entrenamiento
    with open(os.path.join(linkDeGuardado, "autoencoder_summary.txt"), "w") as f:
        autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"batch_size: {bach_size}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"early_stopping: {early_stopping}\n")
        f.write(f"reduce_lr: {reduce_lr}\n")
        f.write(f"optimizer: {optim}\n")
        f.write(f"loss: {'binary_crossentropy'}\n")
        f.write("\n\n")
        f.write(f"Training data shape: {x_train.shape}\n")
        f.write(f"Validation data shape: {x_validation.shape}\n")
        f.write(f"Test data shape: {x_test.shape}\n")
        f.write("\n\n")
        "decoder",decoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        "encoder",encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")

    
     

    encoder.save(linkDeGuardado + "/Encoder.h5")
    decoder.save(linkDeGuardado + "/Decoder.h5")

    #############################################################

    #print("xtest",x_test.shape)
    #latent= encoder.predict(x_test)
    #recontructed_imgs= decoder.predict(latent)
    #img_og=x_test[0]
    #img_reconstructed=recontructed_imgs[0]
    #print('img reconstructed 0',img_reconstructed.shape)
    ##img_og=np.squeeze(img_og)
    ##img_reconstructed=np.squeeze(img_reconstructed)
    ## cargar las imágenes originales y reconstruidas
    #print("reconstruida shape",recontructed_imgs.shape)
    #print ("latent",latent.shape)
#
#
    ##save the data
    #np.save(linkDeGuardado + "/LatentSpace.npy", latent)
    #np.save(linkDeGuardado + "/ReconstructedImages.npy", recontructed_imgs)

    ############################################################################################################################################
    '''
    Empieza la estimacion
    '''
    def add_last(data, new_vals):
        print(f"data: {data.shape} y new_val: {new_vals.shape}")
        x_test_new = data[:,1:]
        print(f"x_test_new: {x_test_new.shape}")

        l = []
        for i in range(len(x_test_new)):
            l.append(np.append(x_test_new[i], new_vals[i]))
        x_test_new = np.array(l).reshape(data.shape[:])
        print("CX", x_test_new.shape)
        return x_test_new
    
    def create_shifted_frames_2(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, data.shape[1]-1, :, :]
        return x, y

    a = np.load("/media/mccdual2080/Almacenamiengto/SahirProjects/SahirReyes/dataSetAutoencoder/DatasetAutoencoder/DataSetLatentSpace/Npy/Balanced/V1/Dataset120x360GreysNewCategories.npy")
    a = a/255
    np.save(linkDeGuardado + "/x_test_autoencoder.npy", a)
    print("a shape",a.shape)
    print("a dtype",a.dtype)
    print("a max",a.max())
    print("a min",a.min())

    latent = encoder.predict(a)
    print("latent shape",latent.shape)
    print("latent dtype",latent.dtype)
    print("latent max",latent.max())
    print("latent min",latent.min())

    np.save(linkDeGuardado + "/LatentSpace.npy", latent)

    channels = 1
    window = 10
    categories = [0, 35, 70, 119, 177, 220, 255] 
    horizon = 4
    imagenInicial = 300

    parte = "EspacioLatente"

    x = np.load(linkDeGuardado + "/LatentSpace.npy")


    rows = x.shape[1]
    cols = x.shape[2]
    print("rows",rows)
    print("cols",cols)    
    
    print("Parte", parte)
    print("x", x.shape)
    print("x", x.dtype)
    print("x", x.min())
    print("x", x.max())


    x_2 = agroup_window(x, window)
    print(x_2.shape)
    x_train = x_2[:int(len(x_2)*.7)]
    x_test = x_2[int(len(x_2)*.7):]
    x_validation = x_train[int(len(x_train)*.8):]
    x_train = x_train[:int(len(x_train)*.8)]

    x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
    x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
    x_test = x_test.reshape(len(x_test), window, rows, cols, channels)

    print("Forma de datos de entrenamiento: {}".format(x_train.shape))
    print("Forma de datos de validación: {}".format(x_validation.shape))
    print("Forma de datos de pruebas: {}".format(x_test.shape))

    x_train, y_train = create_shifted_frames_2(x_train)
    x_validation, y_validation = create_shifted_frames_2(x_validation)
    x_test, y_test = create_shifted_frames_2(x_test)

    print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
    print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
    print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

    np.save(linkDeGuardado+"/x_test_mask.npy", x_test)
    np.save(linkDeGuardado+"/y_test_mask.npy", y_test)
    np.save(linkDeGuardado+"/x_train_mask.npy", x_train)
    np.save(linkDeGuardado+"/y_train_mask.npy", y_train)
    np.save(linkDeGuardado+"/x_validation_mask.npy", x_validation)
    np.save(linkDeGuardado+"/y_validation_mask.npy", y_validation)


    # Define the path where you want to save the log file
    log_file_path = linkDeGuardado+"/InfoConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".txt"

    # Save the original stdout so we can restore it later
    original_stdout = sys.stdout

    #Construction of Convolutional LSTM network
    inp = keras.layers.Input(shape=(None, *x_train.shape[2:]))
    #It will be constructed a 3 ConvLSTM2D layers with batch normalization,
    #Followed by a Conv3D layer for the spatiotemporal outputs.
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    model = keras.models.Model(inp, m)
    model.compile(loss= "binary_crossentropy", optimizer= "Adam")
    print(model.summary())
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 6, restore_best_weights= True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 6)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= linkDeGuardado+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5",
        monitor= "val_loss",
        save_best_only= True,
        mode= "min"
    )
    # Model training with logs redirected to a file
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file  # Redirect stdout to the log file
        model.fit(
            x_train, y_train,
            batch_size=2,
            epochs=50,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stopping, reduce_lr]
        )
        sys.stdout = original_stdout  # Restore stdout back to normal

    print(f"Training log was saved to {log_file_path}")

    #Guardar el modelo
    
    model.save(linkDeGuardado+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5")


    print(imagenInicial)

    example = x_test[imagenInicial]

    print(example.shape)

    err = model.evaluate(x_test, y_test, batch_size= 2)
    print("El error del modelo es: {}".format(err))
    preds = model.predict(x_test, batch_size= 2)
    print("preds",preds.shape)
    x_test_new = add_last(x_test, preds[:])
    preds2 = model.predict(x_test_new, batch_size= 2)
    print("preds2",preds2.shape)
    x_test_new = add_last(x_test_new, preds2[:])
    preds3 = model.predict(x_test_new, batch_size= 2)
    print ("preds3",preds3.shape)
    x_test_new = add_last(x_test_new, preds3[:])
    preds4 = model.predict(x_test_new, batch_size= 2)
    print ("preds4",preds4.shape)
    res_forecast = add_last(x_test_new, preds4[:])
    print("PREDSS",res_forecast.shape)

    np.save(linkDeGuardado+"/PredictionsConvolutionLSTM_forecast_"+str(rows)+"_"+str(cols)+"_"+parte+"_w"+str(window)+".npy", res_forecast)  #Guardar el vector de predicciones

    print("Res_forecast" , res_forecast.shape)

    print("x_test" , x_test.shape)
    print("x_test_new" , x_test_new.shape)
    print("y_test" , y_test.shape)

    #############################################################################################################################################
    '''
    Empieza la decodificación
    '''
    data = np.load(linkDeGuardado+"/PredictionsConvolutionLSTM_forecast_"+str(rows)+"_"+str(cols)+"_"+parte+"_w"+str(window)+".npy")

    results = np.zeros((374, 4, 120, 360, 1))

    for i in range(data.shape[0]):
        # Selecciona los últimos 4 marcos de cada muestra
        last_4_frames = data[i, -4:, :, :, :]
        #datas255 = data[i, :, :, :, :]
        # Realiza la predicción utilizando los últimos 4 marcos
        result = decoder.predict(last_4_frames)

        # Guarda el resultado en el arreglo de resultados
        results[i] = result
        print("result N°",i, "shape", result.shape)


    #guardar resultado
    np.save(linkDeGuardado+"/resultadosDecoder.npy", results)


    #############################################################################################################################################
    '''
    Empieza la parte de la matriz de confusión
    
    # matriz de confusion 
    data = recontructed_imgs

    x_test = x_test

    y_test = x_test



    #x_test = np.load("DataSetLatentSpace/Models/AutoencoderBalanced/Part0_0/x_test_mask.npy")
    #y_test = np.load("DataSetLatentSpace/Models/AutoencoderBalanced/Part0_0/y_test_mask.npy")

    classes = np.array([0, 255, 220, 177, 119, 70, 35]) # 255, 220, 177, 119, 70, 35  0
    classesBalanced = np.array([ 234, 198, 162, 126, 90, 54, 18]) 
    
    rows = 120
    cols= 360
    print(rows)
    print(cols)
    h = 4

    print(data.shape)
    print(x_test.shape)
    print(y_test.shape)
    print("data dtype",data.dtype)
    print("x_test dtype",x_test.dtype)
    print("y_test dtype",y_test.dtype)

    colors = get_colors(x_test[-10])
    print("COLORSS", colors)
    print("COLORS", colors.shape)
    
    colorss = get_colors_optimized(data[-10])
    print("COLORSS", colorss)
    
    y_test = y_test 
    naive = x_test
    new_data = data
    #y_real = y_test[:, -h:]*255
    #new_data = data
    n_real = naive*255
    
    #y_test = y_test[:, -h:]
    #naive = naive
    print("naive shape",naive.shape)
    print("new_data shape",new_data.shape)
    print("n real shape",n_real.shape)
    
    print("XX")
    print(y_test.shape)
    print(new_data.shape)
    print(n_real.shape)
    
    print(min(new_data[0,0,60]))
    print(max(new_data[0,0,60]))
    
    new_data = new_data * 255
    new_data = new_data.astype(np.uint8)
    
    print("new_data", new_data.shape)
    print(colorss.shape)
    print(min(new_data[0,0,60]))
    print(max(new_data[0,0,60]))
    
    new_data = new_data.reshape(new_data.shape[:-1])
    print("HoY", new_data.shape)
    
    
    
    aux = []
    for img in new_data:
        # Convertir la imagen a la escala de grises cuantizada según 'categories'
        res = gray_quantized_optimized(img, categoriesBalanced)
        # Recolorar la imagen cuantizada según 'categories'
        res = recolor_greys_image_optimized(res, categoriesBalanced)
        # Añadir la imagen procesada a 'aux2'
        aux.append(res)
    
    
    new_data = np.array(aux)
    print("SHAPEE", new_data.shape)
    
    
    
    color_data = get_colors(new_data[-10])
    print("DCOLORS", color_data)
    
    
    y_test = y_test * 255
    naive = naive * 255
    
    print("YCOLORS", get_colors(y_test[-10]))
    print("NCOLORS", get_colors(naive[-10]))
    print("DCOLORS", get_colors(new_data[-10]))
    
    print("XS")
    print(f"new data shape {new_data.shape}")
    print(f"y_test.shape {y_test.shape}")
    print(f"new data shape {naive.shape}")
    
    rango = range(y_test.shape[0])
    rango = list(rango)
    
    #print("RANGO", rango)
    
    l_clas = len(classesBalanced)
    rows = 120
    cols= 360
    #print 
    print (f"lengeth x_test: {y_test.shape[0]}")
    print (f"h: {h}")
    print (f"rows: {rows}")
    print (f"cols: {cols}")
    
    cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    #print(cm_f)
    
    for e in rango:
        print(f"e: {e}")
        for i in range(rows):
            for j in range(cols):
                # Identificar la posición de la clase verdadera y las predichas en las clases balanceadas
                #print(f"e: {e}, i: {i}, j: {j}")
                pos1 = np.where(classesBalanced == y_test[e, i, j])[0][0]
                pos2 = np.where(classesBalanced == new_data[e, i, j])[0][0]
                pos3 = np.where(classesBalanced == naive[e, i, j])[0][0]
                # Actualizar las matrices de confusión
                cm_f[pos1, pos2] += 1
                cm_n[pos1, pos3] += 1
    
    print("Matriz de confusión de pronóstico")
    print(cm_f)
    print("Matriz de confusión de naive")
    print(cm_n)
    
    
    
    # Convert cm_f numpy array to pandas DataFrame
    df_cm_f = pd.DataFrame(cm_f)
    
    #print(df_cm_f)
    
    df_cm_n = pd.DataFrame(cm_n)
    
    #print(df_cm_n)
    
    # Crear el DataFrame de la primera matriz de confusión como antes
    df_cm_f = pd.DataFrame(cm_f, index=[f'True_{i}' for i in range(len(cm_f))],
                           columns=[f'Pred_{i}' for i in range(len(cm_f[0]))])
    
    # Crear el DataFrame de la segunda matriz de confusión como antes
    df_cm_n = pd.DataFrame(cm_n, index=[f'True_{i}' for i in range(len(cm_n))],
                               columns=[f'Pred_{i}' for i in range(len(cm_n[0]))])
    
    # Calcular el desplazamiento necesario para la segunda matriz (longitud de la primera matriz + 2 por la columna vacía)
    offset = df_cm_f.shape[1] + 2
    
    # Crear un escritor de Excel
    with pd.ExcelWriter(linkDeGuardado+"/combined_confusion_matrices.xlsx") as writer:
        # Escribir la primera matriz en la hoja de cálculo empezando en la primera columna
        df_cm_f.to_excel(writer, startcol=0, index=True)
    
        # Escribir la segunda matriz en la hoja de cálculo con un desplazamiento
        df_cm_n.to_excel(writer, startcol=offset, index=True)
    
        #with pd.ExcelWriter("DroughtDatasetMask/NPY61_180"+carpeta+"/cm_f_n.xlsx") as writer:
        #    df_cm_f.to_excel(writer, sheet_name='cm_f')
        #    df_cm_n.to_excel(writer, sheet_name='cm_n')
        '''