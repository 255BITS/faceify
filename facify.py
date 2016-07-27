import sys, math, Image
import os
import os.path
import numpy as np
import cv,cv2

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=(np.array(eye_left)+np.array(eye_right))/2, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image




def get_eyes(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = np.asarray(image)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(20,20), maxSize=(2000,2000))

    for (x,y,w,h) in faces:
      roi_gray = gray[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(20,20), maxSize=(400,400))

      def eye_loc(eye):
          #return eye[0]+x+(eye[2]/2), eye[1]+y+(eye[3]/2)
          return eye[0]+x+(eye[2]/2), eye[1]+y+(eye[3]/2)
      
      if len(eyes) < 2:
          return None
      eyes = sorted(eyes, key=lambda eye: eye[3]*eye[1])
      eyes = eyes[0:2]
      eyes = sorted(eyes, key=lambda eye: eye[0])
      #cv2.circle(img, eye_loc(eyes[0]), 20, 0)
      #cv2.circle(img, eye_loc(eyes[1]), 20, 255)
      #cv2.imshow('img',img)
      #cv2.waitKey(0)
      #return [eye_loc(eyes[1]), eye_loc(eyes[0])]
      return [eye_loc(eyes[0]), eye_loc(eyes[1])]

good=0
bad=0
for file in os.listdir("original"):
    name = file.split(".")
    if(len(name) > 1):
        name = name[-2]
    else:
        name = name[-1]

    destination = "crop_faces/"+name + ".jpg"
    if os.path.isfile(destination):
        print("Skip", destination)
        continue
    input_file = "original/"+file
    try:
        image = Image.open(input_file)
    except:
        print("Skipping ", input_file)
        continue
    eyes = get_eyes(image)
    if eyes:
        offset = (0.4,0.4)
        try:
            CropFace(image, eye_left=eyes[0], eye_right=eyes[1], offset_pct=offset, dest_sz=(256,256)).save(destination)
            good+=1
        except:
            print("Could not save", destination)
            bad+=1
    else:
        bad+=1
        print("Skipping", name)

    print("Good/bad ratio", float(good)/(bad+good))
