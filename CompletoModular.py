import os
import time
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image



def monitor_resources(output_file='resource_monitor.xlsx'):
    # Lista para almacenar los datos
    data = []
    # Medir uso de CPU
    cpu_usage = psutil.cpu_percent(interval=0)

    # Medir uso de memoria
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent

        # Medir uso de GPU
    gpu_data = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_data.append({
            'gpu_id': gpu.id,
            'gpu_name': gpu.name,
            'gpu_usage': gpu.load * 100,
            'gpu_memory_usage': gpu.memoryUtil * 100,
            'gpu_memory_total': gpu.memoryTotal,
            'gpu_memory_free': gpu.memoryFree,
            'gpu_memory_used': gpu.memoryUsed
        })

    # Almacenar los datos en la lista
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    for gpu in gpu_data:
       data = np.append(data, {
        'timestamp': timestamp,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        **gpu
        })
       

    print("Monitoreo detenido. Guardando datos...")

    # Crear un DataFrame de pandas
    df = pd.DataFrame(data)

    # Guardar los datos en un archivo Excel
    df.to_excel(output_file, index=False)
    print(f"Datos guardados en {output_file}")

# Función para detener el monitoreo
def stop_monitoring_resources():
    global stop_monitoring
    stop_monitoring = True

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

def get_user_input():
    carpeta = input("Ingrese el nombre de la carpeta: ")
    print(carpeta)
    return carpeta

def create_folder_if_not_exists(linkDeGuardado, carpeta):
    if not os.path.exists(linkDeGuardado + carpeta):
        os.makedirs(linkDeGuardado + carpeta)
    else:
        print("La carpeta ya existe")
    return linkDeGuardado + carpeta

def start_monitoring(monitor_resources):
    monitor_thread = threading.Thread(target=monitor_resources, args=(1, 'resource_monitor.xlsx'))
    monitor_thread.start()

def setup_strategy():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    return strategy

def load_data():
    x_load = np.load("/media/mccdual2080/Almacenamiengto/SahirProjects/SahirReyes/dataSetAutoencoder/DatasetAutoencoder/DataSetLatentSpace/Npy/Balanced/V1/Dataset120x360GreysNewCategories.npy")
    return x_load / 255

def split_data(x):
    x_train = x[:int(len(x) * .7)]
    x_test = x[int(len(x) * .7):]
    x_validation = x_train[int(len(x_train) * .8):]
    x_train = x_train[:int(len(x_train) * .8)]
    return x_train, x_validation, x_test

def build_autoencoder(input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    encoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    decoder_input = keras.Input(shape=encoder_output.shape[1:])
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(decoder_input)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    autoencoder_input = keras.Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")
    return encoder, decoder, autoencoder

def train_autoencoder(autoencoder, x_train, x_validation, linkDeGuardado, batch_size, epochs, learning_rate, patience):
    optim = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optim, loss="binary_crossentropy")

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience)

    history = autoencoder.fit(x=x_train, y=x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, x_validation), callbacks=[early_stopping, reduce_lr])

    save_training_history(history, linkDeGuardado)
    save_model_summary(autoencoder, linkDeGuardado, batch_size, epochs, learning_rate, patience, early_stopping, reduce_lr, optim, autoencoder.layers[1], autoencoder.layers[2], x_train, x_validation, x_train)

    return autoencoder

def save_training_history(history, linkDeGuardado):
    with open(os.path.join(linkDeGuardado, "training_history.txt"), "w") as f:
        f.write("Loss\n")
        f.write(str(history.history['loss']) + '\n')
        f.write("Validation Loss\n")
        f.write(str(history.history['val_loss']) + '\n')
        f.write("Learning Rate\n")
        f.write(str(history.history['lr']) + '\n')
        f.write("Epochs\n")
        f.write(str(history.epoch) + '\n')

def save_model_summary(model, linkDeGuardado, batch_size, epochs, learning_rate, patience, early_stopping, reduce_lr, optim,encoder,decoder,x_train,x_validation,x_test):
    with open(os.path.join(linkDeGuardado, "autoencoder_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"batch_size: {batch_size}\n")
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
        f.write("\n\n")

def save_encoder_decoder(encoder, decoder, linkDeGuardado):
    encoder.save(linkDeGuardado + "/Encoder.h5")
    decoder.save(linkDeGuardado + "/Decoder.h5")

def load_and_normalize_data(path):
    data = np.load(path)
    return data / 255

def save_latent_space(encoder, data, linkDeGuardado):
    latent = encoder.predict(data)
    np.save(linkDeGuardado + "/LatentSpace.npy", latent)
    return latent

def reshape_data(x, window, rows, cols, channels):
    print("x: ", x.shape)
    print("window: ", window)
    print("rows: ", rows)
    print("cols: ", cols)
    x = x.reshape(len(x), window, rows, cols, channels)
    
    return x
    
def append_channel(x,channel, axis):
    x = np.append(x, values=channel, axis=axis)
    return x

def create_shifted_frames(x):
    y = np.copy(x)
    x = x[:, :-1, :, :, :]
    y = y[:, 1:, :, :, :]
    return x, y

def create_conv_lstm_model(rows, cols, channels):
    inp = keras.layers.Input(shape=(None, rows, cols, channels))
    m = keras.layers.ConvLSTM2D(16, (5, 5), padding="same", return_sequences=True, activation="relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5, 5), padding="same", return_sequences=True, activation="relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3, 3), padding="same", activation="relu")(m)
    m = keras.layers.Conv2D(channels, (3, 3), activation="sigmoid", padding="same")(m)
    model = keras.models.Model(inp, m)
    return model

def build_and_train_conv_lstm(x_train, y_train, x_validation, y_validation, linkDeGuardado, rows, cols, channels,  batch_size, epochs, patience):
    model = create_conv_lstm_model(rows, cols, channels)
    model.compile(loss="binary_crossentropy", optimizer="Adam")

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience= patience, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience= patience )
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=linkDeGuardado + "/ConvLSTM2D_Mask.h5", monitor="val_loss", save_best_only=True, mode="min")

    callbacks = [early_stopping, reduce_lr, model_checkpoint]

    model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_validation, y_validation), callbacks=callbacks)

    #model.save(linkDeGuardado + "/ConvLSTM2D_Mask.h5")
    save_model_info_txt(model, linkDeGuardado, rows, cols, channels, batch_size, epochs, patience)

    return model

def save_model_info_txt(model, linkDeGuardado, rows, cols, channels, batch_size, epochs, patience):
    with open(linkDeGuardado + "/ConvLSTM2D_Mask_architecture.txt", "w") as txt_file:
        txt_file.write("Modelo: ConvLSTM2D\n")
        txt_file.write("Hiperparámetros:\n")
        txt_file.write(f"Filas: {rows}\n")
        txt_file.write(f"Columnas: {cols}\n")
        txt_file.write(f"Canales: {channels}\n")
        txt_file.write(f"Batch Size: {batch_size}\n")
        txt_file.write(f"Epochs: {epochs}\n")
        txt_file.write(f"Patience: {patience}\n")
        txt_file.write("\nArquitectura del modelo:\n")
        model.summary(print_fn=lambda x: txt_file.write(x + '\n'))

def evaluate_and_forecast(model, x_test, y_test, linkDeGuardado):
    err = model.evaluate(x_test, y_test, batch_size=2)
    print("El error del modelo es: {}".format(err))
    preds = model.predict(x_test, batch_size=2)
    x_test_new = add_last(x_test, preds)
    preds2 = model.predict(x_test_new, batch_size=2)
    x_test_new = add_last(x_test_new, preds2)
    preds3 = model.predict(x_test_new, batch_size=2)
    x_test_new = add_last(x_test_new, preds3)
    preds4 = model.predict(x_test_new, batch_size=2)
    res_forecast = add_last(x_test_new, preds4)
    np.save(linkDeGuardado + "/PredictionsConvolutionLSTM_forecast.npy", res_forecast)
    return res_forecast

def decode_predictions(decoder, data, linkDeGuardado):
    results = np.zeros((data.shape[0], 4, 120, 360, 1))
    for i in range(data.shape[0]):
        last_4_frames = data[i, -4:, :, :, :]
        result = decoder.predict(last_4_frames)
        results[i] = result
    np.save(linkDeGuardado + "/resultadosDecoder.npy", results)
    return results

def categorize_image(new_data, classesBalanced):
    new_data = new_data * 255
    new_data = new_data.astype(np.uint8)
    print("new_data: ", new_data.shape)
    new_data = new_data.reshape(new_data.shape[:-1])
    print("new_datav2: ", new_data.shape)
    aux = []
    for i in new_data:
        aux2 = []
        for j in i:
            #res = cv2.cvtColor(j, cv2.COLOR_GRAY2RGB)
            #res = recolor_greys_image(j, classes)
            #rgb_quantized(res, classes_rgb)
            #res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            res = gray_quantized_optimized(j, classesBalanced)
            res = recolor_greys_image_optimized(res, classesBalanced)
            aux2.append(res)
        aux.append(np.array(aux2))
    new_data = np.array(aux)
    new_data = new_data.reshape(new_data.shape[0],new_data.shape[1],new_data.shape[2],new_data.shape[3],1)
    return new_data

def save_img(nombre,data , linkDeGuardado, classesBalanced, h):
        
        l_clas = len(classesBalanced)

        fig = plt.figure(figsize=(10,10))
        r = 3
        c = 4
        ac = 1
        pos = 100
        for i in range(h):
            #fig.add_subplot(r, c, ac)
            ac += 1
            plt.imshow(data[pos,i], cmap='gray')
            plt.axis('off')
            plt.title(nombre+'_t+{}'.format(i+1))
            plt.savefig(linkDeGuardado + "/"+nombre+"_t+{}.jpg".format(i+1))

def calculate_confusion_matrix(new_data, y_test, naive, classesBalanced, rows, cols, horizon):
    cm_f = np.zeros((len(classesBalanced), len(classesBalanced)), dtype=np.uint64)
    cm_n = np.zeros((len(classesBalanced), len(classesBalanced)), dtype=np.uint64)
    print("cm_f: ", cm_f.shape)
    print("cm_n: ", cm_n.shape)
    y_test = y_test * 255
    naive = naive * 255
    print("valores unicos de y_test: ", np.unique(y_test))
    print("valores unicos de naive: ", np.unique(naive))
    print("valores unicos de new_data: ", np.unique(new_data))
    for e in range(y_test.shape[0]):
        for k in range(horizon):
            for i in range(rows):
                for j in range(cols):
                    #print(f"e: {e}, k: {k}, i: {i}, j: {j}")
                    
                    pos1 = np.where(classesBalanced == y_test[e, k, i, j])[0][0]
                    pos2 = np.where(classesBalanced == new_data[e, k, i, j])[0][0]
                    pos3 = np.where(classesBalanced == naive[e, k, i, j])[0][0]
                    cm_f[pos1, pos2] += 1
                    cm_n[pos1, pos3] += 1
        print(f"e: {e}")
    return cm_f, cm_n

def save_confusion_matrices_to_excel(cm_f, cm_n, linkDeGuardado):
    df_cm_f = pd.DataFrame(cm_f, index=[f'True_{i}' for i in range(len(cm_f))], columns=[f'Pred_{i}' for i in range(len(cm_f[0]))])
    df_cm_n = pd.DataFrame(cm_n, index=[f'True_{i}' for i in range(len(cm_n))], columns=[f'Pred_{i}' for i in range(len(cm_n[0]))])
    offset = df_cm_f.shape[1] + 2

    with pd.ExcelWriter(linkDeGuardado + "/combined_confusion_matrices.xlsx") as writer:
        df_cm_f.to_excel(writer, startcol=0, index=True)
        df_cm_n.to_excel(writer, startcol=offset, index=True)

def define_rows_cols(data):
    if len(data.shape) == 4:
        rows = data.shape[1]
        print("rows: ", rows)
        cols = data.shape[2]
        print("cols: ", cols)
        return rows, cols
    elif len(data.shape) == 5:
        rows = data.shape[2]
        print("rows: ", rows)
        cols = data.shape[3]
        print("cols: ", cols)
        return rows, cols
    return 0, 0

def displaceData(x_test, data):
    naive = x_test[:-4]
    data = data[1:-3]
    return naive, data

def naive_window(naive, h):
    n_real = naive[:, -h:]
    
    naive = naive[:, -h:]
    return n_real, naive

def monitor_resources(output_file='resource_monitor.xlsx'):
    # Lista para almacenar los datos
    data = []
    # Medir uso de CPU
    cpu_usage = psutil.cpu_percent(interval=0)

    # Medir uso de memoria
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent

        # Medir uso de GPU
    gpu_data = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_data.append({
            'gpu_id': gpu.id,
            'gpu_name': gpu.name,
            'gpu_usage': gpu.load * 100,
            'gpu_memory_usage': gpu.memoryUtil * 100,
            'gpu_memory_total': gpu.memoryTotal,
            'gpu_memory_free': gpu.memoryFree,
            'gpu_memory_used': gpu.memoryUsed
        })

    # Almacenar los datos en la lista
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    for gpu in gpu_data:
       data = np.append(data, {
        'timestamp': timestamp,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        **gpu
        })
       

    print("Monitoreo detenido. Guardando datos...")

    # Crear un DataFrame de pandas
    df = pd.DataFrame(data)

    # Guardar los datos en un archivo Excel
    df.to_excel(output_file, index=False)
    print(f"Datos guardados en {output_file}")

def time_monitoring(linkDeGuardado, training_duration , paso_1_duration, paso_2_duration, paso_3_duration, paso_4_duration, paso_5_duration):
    time_log_file_path = os.path.join(linkDeGuardado, "Total_time.txt")
    with open(time_log_file_path, 'w') as log_file:
        log_file.write(f"\nTotal time completed in {training_duration:.2f} seconds.\n")
        log_file.write(f"\nPaso 1 completed in {paso_1_duration:.2f} seconds.\n")
        log_file.write(f"\nPaso 2 completed in {paso_2_duration:.2f} seconds.\n")
        log_file.write(f"\nPaso 3 completed in {paso_3_duration:.2f} seconds.\n")
        log_file.write(f"\nPaso 4 completed in {paso_4_duration:.2f} seconds.\n")
        log_file.write(f"\nPaso 5 completed in {paso_5_duration:.2f} seconds.\n")

    
def start_time_monitoring():
    start_time = time.time()
    return start_time

def end_time_monitoring(start_time):
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")
    return training_duration

def paso_1(input_shape, bach_size, epochs, learning_rate, patience, linkDeGuardado):
    
    """
    #Paso 1
    """
    paso_1_start_time = time.time()
    print("\nPaso 1\n")
    #carga de data
    x_load = load_data()
    print("x_load: ", x_load.shape)
    print("x_load min: ", x_load.min())
    print("x_load max: ", x_load.max())
    #separacion de data
    x_train, x_validation, x_test = split_data(x_load)
    print("x_train: ", x_train.shape)
    print("x_validation: ", x_validation.shape)
    print("x_test: ", x_test.shape)
    #Arquitectura autoencoder
    encoder, decoder, autoencoder = build_autoencoder(input_shape)
    #entrenamiento autoencoder
    autoencoder = train_autoencoder(autoencoder, x_train, x_validation, linkDeGuardado, bach_size, epochs, learning_rate, patience)
    #guardar encoder y decoder
    save_encoder_decoder(encoder, decoder, linkDeGuardado)
    paso_1_duration = time.time() - paso_1_start_time
    print(f"\nPaso 1 completed in {paso_1_duration:.2f} seconds.")

    return encoder, decoder ,x_load, paso_1_duration

def paso_2(encoder, x_load, linkDeGuardado):
    """
    #Paso 2
    """
    paso_2_start_time = time.time()
    print("\n Paso 2 \n")
    #guardar espacio latente
    latent = save_latent_space(encoder, x_load, linkDeGuardado)
    paso_2_duration = time.time() - paso_2_start_time
    return latent, paso_2_duration

def paso_3( latent, window, rows, cols, channels, bach_size, epochs, patience, linkDeGuardado):
    """
    #Paso 3
    """
    paso_3_start_time = time.time()
    print("\nPaso 3 \n")
    #agrupar en ventanas
    x_2 = agroup_window(latent, window)

    #definir rows y cols
    rows, cols = define_rows_cols(latent)

    #separar data para entrenar ConvLSTM
    x_train, x_validation, x_test = split_data(x_2)
        
    #reshape de data
    x_train = reshape_data(x_train, window, rows, cols, channels)
    x_validation = reshape_data(x_validation, window, rows, cols, channels)
    x_test = reshape_data(x_test, window, rows, cols, channels)

    #crear desplazamientos para convLSTM
    x_train, y_train = create_shifted_frames_2(x_train)
    x_validation, y_validation = create_shifted_frames_2(x_validation)
    x_test, y_test = create_shifted_frames_2(x_test)

    #entrenar 
    model = build_and_train_conv_lstm(x_train, y_train, x_validation, y_validation, linkDeGuardado, rows, cols, channels , bach_size, epochs, patience)
    res_forescast = evaluate_and_forecast(model, x_test, y_test, linkDeGuardado )
    paso_3_duration = time.time() - paso_3_start_time
    return res_forescast, paso_3_duration

def paso_4( decoder, res_forescast, linkDeGuardado, classesBalanced, horizon):
    """
    #Paso 4
    """
    paso_4_start_time = time.time()
    print("\n Paso 4\n ")
    decoded_data = decode_predictions(decoder, res_forescast, linkDeGuardado)
    save_img("decoded_data",decoded_data, linkDeGuardado, classesBalanced, horizon)
    paso_4_duration = time.time() - paso_4_start_time
    return decoded_data, paso_4_duration


def paso_5( decoded_data, window, classesBalanced, horizon, linkDeGuardado, channels):
    """
    Paso 5
    """
    paso_5_start_time = time.time()
    print("\nPaso 5\n")
    #carga de data
    x_load = load_data()
    print("x_load: ", x_load.shape)

    x_2 = agroup_window(x_load, window)
    #separacion de data
    x_train, x_validation, x_test = split_data(x_2)
    print("x_train: ", x_train.shape)
    print("x_validation: ", x_validation.shape)
    print("x_test: ", x_test.shape)
    
    rows, cols = define_rows_cols(decoded_data)

    #reshape de data
    x_train = reshape_data(x_train, window, rows, cols, channels)
    print("x_train: ", x_train.shape)
    x_validation = reshape_data(x_validation, window, rows, cols, channels)
    print("x_validation: ", x_validation.shape)
    x_test = reshape_data(x_test, window, rows, cols, channels)
    print("x_test: ", x_test.shape)

    #crear desplazamientos para convLSTM
    x_train, y_train = create_shifted_frames_2(x_train)
    x_validation, y_validation = create_shifted_frames_2(x_validation)
    x_test, y_test = create_shifted_frames_2(x_test)
        
    print("ytest", y_test.shape)

    y_test = get_cubes(y_test, horizon)
        
    naive, decoded_data = displaceData(x_test, decoded_data)
    n_real, naive = naive_window(naive, horizon)
    new_data = categorize_image(decoded_data, classesBalanced)
    print("valores unicos de new_data: ", np.unique(new_data))
    print("valores unicos de y_test: ", np.unique(y_test))
    print("valores unicos de naive: ", np.unique(naive))

    save_img("new data",new_data, linkDeGuardado, classesBalanced, horizon)
    save_img("y_test",y_test, linkDeGuardado, classesBalanced, horizon)
    save_img("naive",naive, linkDeGuardado, classesBalanced, horizon)

    print("new_data: ", new_data.shape)
    print("y_test: ", y_test.shape)
    print("naive: ", naive.shape)
    print("classesBalanced: ", classesBalanced)
    print("rows: ", rows)
    print("cols: ", cols)
    print("horizon: ", horizon)
    print("len(classesBalanced)", len(classesBalanced))

        
    cm_f, cm_n = calculate_confusion_matrix(new_data, y_test, naive, classesBalanced, rows, cols, horizon)
    save_confusion_matrices_to_excel(cm_f, cm_n, linkDeGuardado)
    paso_5_duration = time.time() - paso_5_start_time
    return paso_5_duration

def main():
    linkDeGuardado = "Resultados/ResultadoCompletoNewCategories/"
    carpeta = get_user_input()
    linkDeGuardado = create_folder_if_not_exists(linkDeGuardado, carpeta)
    start_total_time = start_time_monitoring()
    start_monitoring(monitor_resources)

    strategy = setup_strategy()

    input_shape = (120, 360, 1)

    bach_size=2
    learning_rate = 0.005
    epochs=2
    patience = 10
    window = 10
    channels = 1
    horizon = 4
    imagenInicial = 300
    classesBalanced = np.array([18, 54, 90, 126, 162, 198, 234])
    classes = np.array([0, 255, 220, 177, 119, 70, 35]) 

    rows = 120
    cols = 360

    with strategy.scope():
        
        encoder, decoder , x_load, paso_1_duration = paso_1(input_shape, bach_size, epochs, learning_rate, patience, linkDeGuardado)
        
        latent , paso_2_duration = paso_2(encoder, x_load, linkDeGuardado)

        res_forescast , paso_3_duration = paso_3(latent, window, rows, cols, channels, bach_size, epochs, patience, linkDeGuardado)

        decoded_data, paso_4_duration = paso_4(decoder, res_forescast, linkDeGuardado, classesBalanced, horizon)

        paso_5_duration = paso_5(decoded_data, window, classesBalanced, horizon, linkDeGuardado, channels)
        
        training_duration = end_time_monitoring(start_total_time)
        time_monitoring(linkDeGuardado, training_duration, paso_1_duration, paso_2_duration, paso_3_duration, paso_4_duration, paso_5_duration)
        stop_monitoring_resources()
        

if __name__ == "__main__":
    main()
