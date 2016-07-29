import sys, math, Image
import os
import os.path
import numpy as np
import cv,cv2

import dlib
import argparse

parser = argparse.ArgumentParser(description='Runs the GAN.')

parser.add_argument('--directory', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()
DIR = args.directory
#DIR = "../data/celeb/"
#DIR = "/home/martyn/Downloads/imdb/imdb/81/"

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
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)

  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))

  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=(np.array(eye_left)+np.array(eye_right))/2, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  if(crop_xy[0] < 0 or crop_xy[1] < 0 or int(crop_xy[0]+crop_size[0]) > image.size[0] or int(crop_xy[1]+crop_size[1]) > image.size[1]):
      print("CROP OVER", crop_xy, crop_size, image.size)
      return None
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image




def get_eyes(image):
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    img = np.array(image)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None


    dets = detector(gray, 1)
    faces = []
    for k,d in enumerate(dets):
        shape = predictor(gray, d)
        s1 = shape.part(40)
        s2 = shape.part(47)
        l = np.array((int(s1.x),s1.y))
        r = np.array((int(s2.x),s2.y))
        threshold = 60
        if(abs((r-l)[0])<threshold):
          continue
        print("appending face")
        faces.append([l,r])

    print('--')
    return faces

good=0
bad=0
for file in os.listdir(DIR):
    name = file.split(".")
    if(len(name) > 1):
        name = name[-2]
    else:
        name = name[-1]

    destination = args.output+"/"+name + ".jpg"
    if os.path.isfile(destination.split(".")[-2]+".1"+destination.split(".")[-1]):
        print("Skip", destination)
        continue
    input_file = DIR+file
    try:
        image = Image.open(input_file)
    except:
        print("Skipping ", input_file)
        continue
    eyes = get_eyes(image)
    if eyes:
        e_count=0
        for e in eyes:
            e_count+=1
            offset = (0.4,0.51)
            newdestination = destination.split(".")[-2] + "."+str(e_count)+"." + destination.split(".")[-1]
            print("Cropping to "+newdestination)
            res = CropFace(image, eye_left=e[0], eye_right=e[1], offset_pct=offset, dest_sz=(256,256))
            if(res == None):
                bad+=1
            else:
                res.save(newdestination)
                good+=1
    else:
        bad+=1
        print("Skipping", name)

    print("Good/bad ratio", float(good)/(bad+good))
