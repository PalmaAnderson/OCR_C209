# coding: utf-8                     #codificação de acentos

# Anderson Andre Palma
# GEC - 1010

#Instruçoes de instalação:
#pip install pytesseract
#pip install opencv-python
#pip install unidecode
#install tesseract for widonws
#https://github.com/UB-Mannheim/tesseract/wiki


import pytesseract                  #biblioteca principal deste codigo
from unidecode import unidecode     #imprimir caracteres
import numpy as np 
import cv2                          #open cv 2, o "motor grafico"
from PIL import Image   
import argparse
import matplotlib.pyplot as plt
import time




def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped



def edge_detect():
    img_rgb = np.array(Image.open("inclinado.jpg"))[:, :, :3]
    l, c, p = img_rgb.shape
    # converter para escala de cinza:
    img = np.zeros(shape=(l, c), dtype=np.uint8)
    for i in range(l):
        for j in range(c):
            r = float(img_rgb[i, j, 0])
            g = float(img_rgb[i, j, 1])
            b = float(img_rgb[i, j, 2])
            
            img[i, j] = (r + g + b) / 3


    abs_tg_c = np.abs(np.diff(img.astype(np.float64), axis=0, append=255))
    abs_tg_l = np.abs(np.diff(img.astype(np.float64), axis=1, append=255))
    abs_tg = np.sqrt(abs_tg_c ** 2 + abs_tg_l ** 2)
    threshold = 23
    img_border = np.zeros(shape=(l, c), dtype=np.uint8)
    img_border[abs_tg > threshold] = 255

    Image.fromarray(img_border).save('contorno.png')





def perspectiva():
    image = cv2.imread("C:\Users\\aande\Downloads\Workspace\C209\\trab\\inclinado.jpg")
    pts = np.array(eval("[(642,145), (3496,541), (3317,2821), (368,2518)]"), dtype = "float32")

    warped = four_point_transform(image, pts)
    Image.fromarray(warped).save('recorte.png')

    '''
    Pontos para recortar imagem

    ---------------
    |   X  |    Y  |
    ---------------
    |642   |145    |
    |3496  |541    |
    |3317  |2821   |
    |368   |2518   |   
    ---------------

    '''


def ocr(final,file):
    
    imagem = Image.open(file).convert('RGB')

    npimagem = np.asarray(imagem).astype(np.uint8)  
    im = cv2.cvtColor(npimagem, cv2.COLOR_RGB2GRAY) 
    #threshold = 127 para binariza a imagem
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

    binimagem = Image.fromarray(thresh) 
    if(final):
        time.sleep(3)
        Image.fromarray(thresh).save('FINAL.png')
    # chamada ao tesseract OCR por meio de seu wrapper
    phrase = pytesseract.image_to_string(binimagem, lang='por')

    # impressão do resultado com unidecode devido a acentuaçao da lingua portugues
    #print(phrase)
    print(unidecode(phrase))




print ("\n####\ importacao de bibliotecas concluida\n####\n")

#localiza a intalação do tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

print("\n####\nresultado sem aprimoramento\n####\n")
ocr(0,'C:\Users\\aande\Downloads\Workspace\C209\\trab\\inclinado.jpg')  #imagem anterior

print ("\n####\nrealizando a deteccao e borda\n####\n")
edge_detect()

print ("\n####\nRecortando a imagem\n####\n")
perspectiva()

print("\n####\nresultado apos o tratamento de imagem\n####\n")
ocr(1,'C:\Users\\aande\Downloads\Workspace\C209\\trab\\recorte.png')    #imagem aprimorada
