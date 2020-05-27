from PIL import Image

def sepia(image_path:str)->Image:
    img = Image.open("data/"+image_path)
    width, height = img.size

    pixels = img.load()  # create the pixel map

    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)
    img.save("data/"+image_path.split('.')[0]+"SEPIA.jpg")
    return img

def addSepiaToAll():
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir("data") if isfile(join("data", f))]
    for f in images:
        if str(f).find("SEPIA") == -1 :
            sepia(f)



def getPixelMapForModeRGB(img):
    width, height = img.size

    pixels =[[[0,0,0] for _ in range(width)] for _ in range(height)]  # create the pixel map

    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))
            pixels[py][px] = [r, g, b]
    return pixels

def getPixelMapForModeL(img):
    width, height = img.size

    pixels = [[0 for _ in range(width)] for _ in range(height)]  # create the pixel map

    for py in range(height):
        for px in range(width):
            p = img.getpixel((px, py))
            pixels[py][px] = p
    return pixels

def getImagesWithAndWithoutSepia():
    inputs=[]
    outputs=[]
    names=['normal','sepia']
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir("data") if isfile(join("data", f))]
    for f in images:
        img = Image.open("data/" + f)
        pixels = getPixelMapForModeRGB(img)
        inputs.append(pixels)
        if f.find("SEPIA") != -1:
            outputs.append(1)
        else:
            outputs.append(0)

    return inputs,outputs,names

def getImagesHappySad():
    inputs = []
    outputs = []
    names = ['happy', 'sad']
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir("happy") if isfile(join("happy", f))]
    for f in images:
        img = Image.open("happy/" + f)
        pixels = getPixelMapForModeL(img)
        inputs.append(pixels)
        outputs.append(0)
    images = [f for f in listdir("sadness") if isfile(join("sadness", f))]
    for f in images:
        img = Image.open("sadness/" + f)
        pixels = getPixelMapForModeL(img)
        inputs.append(pixels)
        outputs.append(1)
    return inputs,outputs,names