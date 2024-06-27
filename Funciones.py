

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

