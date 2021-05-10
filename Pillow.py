from zipfile import ZipFile

from PIL import Image
from PIL import ImageDraw
import pytesseract
import cv2 as cv
import numpy as np
# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
file_name = "readonly/small_img.zip"
val = input("Search : ") 
images=[]
count=0
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    for info in zip.infolist(): 
         print(info.filename) 
         data=zip.read(info.filename)
         img = cv.imdecode(np.frombuffer(data, np.uint8), 1) 
         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
         pil_img=Image.fromarray(gray,mode="L")   
         content = pytesseract.image_to_string(pil_img)
         if val in content:
                print("Results found in "+info.filename)
                a=0
                b=0
                cv_img_bin=cv.threshold(img,180,255,cv.THRESH_BINARY)[1]
                faces = face_cascade.detectMultiScale(cv_img_bin,1.15)
                contact_sheet=Image.new("L", (100*5,(int(len(faces)/5)+1)*100))
                if len(faces)==0:
                    print("But there were in faces in that file!")
                else:    
                    for x,y,w,h in faces:
                        temp = pil_img.crop((x,y,w+x,y+h))
                        contact_sheet.paste(temp.resize((100,100)),(a, b) )
                        if a+100 == contact_sheet.width:
                            a=0
                            b=b+100
                        else:
                            a=a+100
                    display(contact_sheet) 
        
        
     
