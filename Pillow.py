import os
import zipfile
from IPython.display import display
from PIL import Image
from PIL import ImageDraw
from kraken import pageseg
from kraken import binarization
import pytesseract
import cv2 as cv
import numpy as np
import string
import copy
import math


class Page():

    def __init__(self, file):
        
        #add filename instance variable
        self.name = file.name

        #open image with OpenCV, we're working with files in RGB colorspace
        img_bytes = np.fromstring(file.read(), np.uint8)
        self.image = cv.cvtColor(cv.imdecode(img_bytes, cv.IMREAD_COLOR) , cv.COLOR_BGR2RGB)
          
        #segment the page into image sections & text sections    
        self.image_sections, self.text_sections = self.__segment_page()  
        
        #extract text and detect any faces
        self.page_text = self.__extract_page_text()
        self.faces = self.__dectect_faces()
    
    
    
    def filename(self):
        return self.name
    
    
    def get_image(self):
        return self.image.copy()
    
    
    def get_page_text(self):
        return self.page_text
    
    
    def get_page_faces(self):
        return copy.deepcopy(self.faces)

    
    def search_page(self, search_term):
        if search_term in self.page_text:
            #search term found in page_text
            return {'filename':self.name, 'faces':self.faces}
        else:
            return None
    
    
    #some private helper methods to aid extracting page content.
    def __segment_page(self):
        
        # seperates out text & image sections of newspaper page
        # reducese noise for improved OCR & face detection
        text_sections = []
        image_sections = []
        
        #create greyscale version for processing purposes
        img_gray = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)

        #binarize and invert - makes text white and whitespace black which helps with contouring
        ret2, inverted = cv.threshold(img_gray, 180, 255, cv.THRESH_BINARY_INV)

        #find contours - after inverting contours will outline main text and image blocks on newspaper page
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
        dilated = cv.dilate(inverted, kernel, iterations=9)
        img, contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

        #iterate over contours, assigning bounding boxes to text or images lists
        #use crude/naive filter to tell if contour bounds an image or not
        # if count(color unique values within bound) > 150 then it's probably an image
        # if count(color uniquue values within bound) < 150 then it's probably text
        for contour in contours:
            
            # get rectangle bounding contour
            x,y,w,h = cv.boundingRect(contour)
            
            # Ignore small bouding boxes
            if w >= 150 and h >= 50:

                segment = img_gray[y:y+h,x:x+w]
            
                if len(np.unique(segment)) >= 150:
                    #probably an image section, append bounding box
                    image_sections.append((x,y,w,h))
                else:
                    #probably a text section, append bounding box
                    text_sections.append((x,y,w,h))    
        
        return image_sections, text_sections
    
    
    def __dectect_faces(self):
        
        face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
        page_faces = []

        for rect in self.image_sections:

            # get rect x,y,w,h
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            #slice image
            section_img = self.image[y:y+h,x:x+w]
            
            #convert image section to grayscale
            segment_gray = cv.cvtColor(section_img, cv.COLOR_RGB2GRAY)
            
            #detect faces, scaleFactor 1.18 seems to work best across newspaper page dataset
            faces = face_cascade.detectMultiScale(segment_gray, 1.18, 4)  

            #simple filtering approach to remove noise & false positives
            face_area_accum = 0.0
            n_samples = 0

            #compute average area of detected faces, used to filter out any 'noisy' artefacts
            #e.g. ear-lobes detected as faces
            for x,y,w,h in faces:
                face_area_accum += w*h
                n_samples += 1

            if len(faces) > 0:
                #some faces were detected in this image i
                avg_face_area = face_area_accum/n_samples
                min_face_area = avg_face_area*0.3

            for x,y,w,h in faces:
                if w*h >= min_face_area:
                    #found a face, append bounding box tuple (x,y,w,h) to faces list
                    face_img = section_img[y:y+h,x:x+w]
                    #page_faces.append((x,y,w,h))
                    page_faces.append(face_img)
                    
        return page_faces

        
    def __extract_page_text(self):    
        
        page_text = ''
    
        for rect in self.text_sections:
            
            # get rect x,y,w,h
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            #convert to grayscale & then binarize. OCR works better on binarized version    
            segment_gray = cv.cvtColor(self.image[y:y+h,x:x+w], cv.COLOR_RGB2GRAY)
            thres, segment_binary = cv.threshold(segment_gray,120,255,cv.THRESH_BINARY)   

            #perform O.C.R.
            text = pytesseract.image_to_string(Image.fromarray(segment_binary))

            if text != '':
                #text found, append to page_text
                page_text += text
        
        return page_text
    
    
def create_contact_sheet(images=[]):
    #creates a contact sheet from a list of images
    #each image in images should be numpy array
    #returns contact sheet as PIL Image
    
    #limit image size to 128x128 (x,y)
    thumbnail_size = 128,128
    
    #configure canvas for contact sheet
    #limit contact sheet to five images per row, number of rows determined dynamically.
    images_per_row = 5
    num_rows = math.ceil(len(images)/images_per_row)
    
    #define dimensions of contact sheet canvas
    canvas_width = thumbnail_size[0]*images_per_row
    canvas_height = thumbnail_size[1]*num_rows

    #create canvas for contact sheet, create RGB in case of mix of colour & grayscale images in source files
    contact_sheet=Image.new('RGB', (canvas_width, canvas_height))
    
    #initialize co-ordinates for pasting thumbnails to contact sheet
    x=0
    y=0
    
    #add images to contact sheet, resizing any as necessary
    for img in images:
        
        #create copy so original remains unmodified
        img_copy = copy.deepcopy(img)
        
        #create PIL version of image
        img_copy_PIL = Image.fromarray(img_copy)
        
        # create thumbnail, re-sizes where required     
        img_copy_PIL.thumbnail(thumbnail_size)

        contact_sheet.paste(img_copy_PIL, (x, y) )
        
        # Now we update our X position. If it is going to be the width of the image, then we set it to 0
        # and update Y as well to point to the next "line" of the contact sheet.
        if x+thumbnail_size[0] == contact_sheet.width:
            x=0
            y=y+thumbnail_size[0]
        else:
            x=x+thumbnail_size[0]
    
    return contact_sheet

if __name__=="__main__":
	import time
	zip_path_small = '/home/jovyan/work/readonly/small_img.zip'
	zip_path_full = '/home/jovyan/work/readonly/images.zip'

	pages_small = []

	with zipfile.ZipFile(zip_path_small) as zip_arxiv:     
    		zip_contents = zip_arxiv.infolist()  
    
    	for zc in zip_contents:
        	#load each newspaper page image in zip file into an instance of Page() 
        	file = zip_arxiv.open(zc)
        
        	print('Loading file: {}'.format(file.name))
        	start= time.time()
        	pages_small.append(Page(file))
        	end = time.time()
        	print('....loaded in {} seconds'.format(end - start))
    

    
    
	# perform text search and display search results
	print('\n\n\n')

	search_term = input("Search For: ")
	print('Searching for: {}'.format(search_term))

	for page in pages_small:
    		result = page.search_page(search_term)
    
    	if result is not None:
        	print('Results found in file {}'.format(result['filename']))
        
        	if len(result['faces']) == 0:
            		#no faces on this page
            		print('But there were no faces in that file!')
        	else:
            		#faces on page, display contact sheet
            		display(create_contact_sheet(result['faces']))