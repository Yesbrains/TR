
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import os


def obtenció_de_coordenades(matriucolor,diccionari):
    y, x = np.where(matriucolor > 0)
    for (py, px) in zip(y, x):
        dx = (px - cx)
        dy = (py - cy)
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.acos(dx / r) * 180 / math.pi
        alpha = 180 - theta
        if py < cy:
            angle = theta
        else:
            angle = theta + 2 * alpha
        r = r - rp
        angle = round(angle,0)
        r = round(r,0)
        if r < 0:
            continue
        elif r <= ri - rp:
            diccionari[(r, angle)] = matriucolor[py][px]
        else:
            continue

def calcul_centre_pupila(imatge):
    y, x = np.where(imatge < 1)
    yt = 0
    xt = 0
    cont = 0
    for y, x in zip(y, x):
        yt += y
        xt += x
        cont += 1

    return(yt/cont, xt/cont)

def calcul_radi_pupila(imatge, cpx, cpy):
    y, x = np.where(imatge < 1)
    dyt = 0
    dxt = 0
    cont = 0
    for y, x in zip(y, x):
        dy = y - cpy
        dx = x - cpx
        if dy < 0:
            dy = -dy
        if dx < 0:
            dx = -dx
        dyt += dy
        dxt += dx
        cont += 1

    dytm = dyt / cont
    dxtm = dxt / cont
    return( int(np.sqrt(dytm ** 2 + dxtm ** 2)) )

def calcul_valor_desviacio(imatge):
    img_np = np.array(imatge)
    desviacio =np.std(img_np)
    return desviacio

def binaritzacio(ds,umbral):
    if ds > umbral:
        return 1
    else:
        return 0

ri = 0
rp = 0
count = 0
normal_values = {}
desviacions = []
numcomplet = []
A = {}
new_file = ""

personName= input("Qui ets? (escriu el teu nom sense accents ni caracters especials): ")
dataPath = 'C:/Users/Jose-Administrador/PycharmProjects/TR/data'
personPath= dataPath + "/" + personName

if not os.path.exists(personPath):
  print("Carpeta creada", personPath)
  os.makedirs(personPath)

imag = cv2.imread("ULLBLAU.jpg", 1)

imgpq = cv2.resize(imag, (640,480), 0.5, 0.5, interpolation=cv2.INTER_CUBIC)

imggrey = cv2.cvtColor(imgpq, cv2.COLOR_BGR2GRAY)

imgb = cv2.blur(imggrey, (11, 11))

imgcanny = cv2.Canny(imgb, 30, 50)

mask = np.zeros((480,640), dtype=np.uint8)

VCircle = cv2.HoughCircles(imgcanny, cv2.HOUGH_GRADIENT, 2, 3000, param1=100, param2=30, minRadius=120, maxRadius=140)
if VCircle is not None:
   VCircle = np.int16(np.around(VCircle))

   for pt in VCircle[0, :]:
       x, y, r = pt[0], pt[1], pt[2]
       ri = r
       cx = x
       cy = y

       cv2.circle(mask, (x, y), r, (255), 1)

       cv2.floodFill(mask, None, (x, y), (255))

   img = cv2.bitwise_and(imgpq, imgpq, mask=mask)

else:
    print("modifica els paràmetres")

imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
y,x = np.where(imgg<1)
for (y,x) in zip(y,x):
    imgg[y][x] = 255

_, imgt = cv2.threshold(imgg,15,255,type=cv2.THRESH_BINARY)

imgl = cv2.medianBlur(imgt,29)
imgl = cv2.medianBlur(imgl,29)
imgl = cv2.medianBlur(imgl,29)

imgnew = cv2.bitwise_and(img,img,mask=imgl)

cpy, cpx = calcul_centre_pupila(imgl)
rp = calcul_radi_pupila(imgl, cx, cy)

cv2.imshow("Original", imggrey)
cv2.imshow("OriginalA", imgnew)
print(ri, rp, (ri-rp))

greydef = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

obtenció_de_coordenades(greydef,normal_values)

w = 362
h = ri - rp + 1
Normal = np.zeros((h, w), np.uint8)

for (r, a), v in normal_values.items():
   Normal[int(r)][int(a)] = v

contrast = cv2.equalizeHist(Normal)

plt.imshow(contrast, cmap="gray")
plt.title('Normalització')
plt.show()

h, w = contrast.shape

imgtall = contrast[:, 1:w-1]
h, w = imgtall.shape

for y in range (35):
    A[y] = imgtall[0:h, y*10:y*10+9]

for y in range (35):
    desviacions.append(calcul_valor_desviacio(A[y]))

umbral = np.mean(desviacions)

for ds in zip(desviacions):
    binari = binaritzacio(ds,umbral)
    numcomplet.append(binari)

modbin0 = str(numcomplet).replace("[","")
modbin1 = modbin0.replace("]","")
modbin2 = modbin1.replace(",","")
modbin3 = modbin2.replace(" ","")

for fileName in os.listdir(personPath):
   count += 1

new_file = personPath+"/codibinari_"+personName+str(count)+".txt"

with open(new_file,"w") as file:
    file.write(str(modbin3))

file.close()

print("S'ha guardat el model correctament en la carpeta"+personPath)

cv2.waitKey(0)
cv2.destroyAllWindows()
