import math
import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_adaptive

def LLA2ECEF(latitude, longitude, altitude):
    """
    Method to convert Latitude Longitude Altitude to Earth-Centered, Earth-Fixed system
    """
    equatorial_radius = 6378137.0
    f = 1.0 / 298.257224    #flattening of earth

    cosLat, cosLon, sinLat, sinLon =trigMath(latitude, longitude)    
    a = equatorial_radius + altitude
    e = math.sqrt(f * (2 - f))
    N = a / math.sqrt(1 - e * e * sinLat * sinLat)
    X = (altitude + N) * cosLon * cosLat
    Y = (altitude + N) * cosLon * sinLat
    Z = ((1 - e * e) * N + altitude) * sinLon

    return X, Y, Z 

def trigMath(latitude, longitude):
    cosLat = (math.cos(latitude * math.pi / 180.0))
    sinLat = (math.sin(latitude * math.pi / 180.0))
    cosLon = (math.cos(longitude * math.pi / 180.0))
    sinLon = (math.sin(longitude * math.pi / 180.0))
    return cosLat, cosLon, sinLat, sinLon


def ECEF2ENU(lat, lon, altitude, cosLat, cosLon, sinLat, sinLon):
    """
    Method to convert Earth-Centered, Earth-Fixed System to Earth North Up System
    """
    x1, y1, z1 = LLA2ECEF(45.904144, 11.028454, 227.581900)
    dx = lat - x1;
    dy = lon - y1;
    dz = altitude - z1;
    r = [[-sinLon, cosLon, 0], [-(cosLon * sinLat), -(sinLat * sinLon), cosLat], [cosLon*cosLat, cosLat*sinLon, sinLat]]
    po = [dx,dy,dz]
    penu = np.dot(r, po)
    x = penu[0]
    y = penu[1]
    z = penu[2]
    return x, y, z

def ENU2CC(e, n, u):
    """
    Method to convert Earth North Up System to Camera Coordinates
    """
    Qs, Qx, Qy, Qz  = 0.362114, 0.374050, 0.592222, 0.615007
    a =  (Qs*Qs) + (Qx*Qx) - (Qy*Qy) - (Qz*Qz)
    b =  (2*Qx*Qy) - (2*Qs*Qz)
    c =  (2*Qx*Qz) + (2*Qs*Qy)
    d =  (2*Qx*Qy) + (2*Qs*Qz)
    e1 =  (Qs*Qs) - (Qx*Qx) + (Qy*Qy) - (Qz*Qz)
    f =  (2*Qz*Qy) - (2*Qs*Qx)
    g =  (2*Qx*Qz) - (2*Qs*Qy)
    h =  (2*Qz*Qy) + (2*Qs*Qx)
    i =  (Qs*Qs) - (Qx*Qx) - (Qy*Qy) + (Qz*Qz)
    rq = [[a, b, c], [d,e1,f], [g,h,i]]
    pneu = [n,e,-u]
    cc = np.dot(rq, pneu)
    x = cc[0]
    y = cc[1]
    z = cc[2]
    return x, y, z

def drawMatches(img1, kp1, img2, kp2, matches):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
    angles = []
    for m in matches:
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        angles.append(math.atan((float)(end2[1] - end1[1]) / (end1[0] - end2[0])) * (180 / math.pi))
    
    return(sum(angles) / len(angles))


def readConfigurations():
    pointcloud = open('./final_project_data/final_project_point_cloud.fuse', 'rb')
    camera_params = pd.read_csv("final_project_data/image/camera.config")
    return pointcloud, camera_params

def main():
    pointcloud, camera_params = readConfigurations()
    latitude, longitude, altitude = camera_params.iloc[0,0], camera_params.iloc[0,1], camera_params.iloc[0,2]

    print("Camera parameters and camera quaternion:")
    print(camera_params)

    point_info = []
    point_cloud_display = []
    frontimage = np.zeros((2048,2048), dtype = float)
    backImage = np.zeros((2048,2048), dtype = float)
    rightImage = np.zeros((2048,2048), dtype = float)
    leftImage = np.zeros((2048,2048), dtype = float)

    for line in pointcloud:
        val = line.decode('utf8').strip().split(' ')
        point = []
        x, y, z= LLA2ECEF(float(val[0]), float(val[1]), float(val[2]))
        cosLat, cosLon, sinLat, sinLon = trigMath(float(val[0]), float(val[1]))
        point.append(x)
        point.append(y)
        point.append(z)
        point.append(cosLat)
        point.append(cosLon)
        point.append(sinLat)
        point.append(sinLon)
        point_info.append(point)

    initial = open('./output/pointcloud.obj', 'w')
    penufile = open('./output/penufile.obj', 'w')
    ccfile = open('./output/ccfile.obj', 'w')
    fffile = open('./output/frontface.obj', 'w')
    bffile = open('./output/backface.obj', 'w')
    rffile = open('./output/rightface.obj', 'w')
    lffile = open('./output/leftface.obj', 'w')
   
    x1, y1, z1 = LLA2ECEF(latitude, longitude, altitude)
    cosLat, cosLon, sinLat, sinLon = trigMath(latitude, longitude)

    xe, yn, zu= ECEF2ENU(float(x1), float(y1), float(z1), float(cosLat), float(cosLon), float(sinLat),float(sinLon))
    xc, yc, zc = ENU2CC(xe, yn, zu)

    for point in point_info:
        line = "v " + str(point[0]) + " " + str(point[1]) + " "+ str(point[2])
        initial.write(line)
        initial.write("\n")
        e,n,u = ECEF2ENU(float(point[0]), float(point[1]), float(point[2]), float(point[3]), float(point[4]), float(point[5]),float(point[6]))
        line2 = "v " + str(e) + " " + str(n) + " "+ str(u)
        penufile.write(line2)
        penufile.write("\n")

        cc1, cc2, cc3 = ENU2CC(e,n,u)
        line3 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
        if (cc3 > 0) & (cc3 > abs(cc1)) & (cc3 > abs(cc2)):
            line4 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
            fffile.write(line4)
            fffile.write("\n")
            xi = int(((cc2/cc3)*((2048-1)/2)) + ((2048+1)/2))
            yi=  int(((cc1/cc3)*((2048-1)/2)) + ((2048+1)/2))
            F = cc3-zc;
            frontimage[xi][yi]=255;
        if (cc1 > 0) & (cc1 > abs(cc3)) & (cc1 > abs(cc2)):
            line5 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
            rffile.write(line5)
            rffile.write("\n")
            xi = int(((cc2/cc1)*((2048-1)/2)) + ((2048+1)/2))
            yi=  int(((cc3/cc1)*((2048-1)/2)) + ((2048+1)/2))
            rightImage[xi][yi]=255;
        if (cc3 < 0) & (abs(cc3) > abs(cc1)) & (abs(cc3) > abs(cc2)):
            line6 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
            bffile.write(line6)
            bffile.write("\n")
            xi = int(-((cc2/cc3)*((2048-1)/2)) + ((2048+1)/2))
            yi=  int(-((cc1/cc3)*((2048-1)/2)) + ((2048+1)/2))
            backImage[xi][yi]=255;
        if (cc1 < 0) & (abs(cc1) > abs(cc3)) & (abs(cc1) > abs(cc2)):
            line7 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
            lffile.write(line7)
            lffile.write("\n")
            xi = int(-((cc2/cc1)*((2048-1)/2)) + ((2048+1)/2))
            yi=  int(-((cc3/cc1)*((2048-1)/2)) + ((2048+1)/2))
            leftImage[xi][yi]=255;
        ccfile.write(line3)
        ccfile.write("\n")

    cv2.imwrite('./output/front.png',frontimage)
    cv2.imwrite('./output/back.png',backImage)
    cv2.imwrite('./output/right.png',rightImage)
    cv2.imwrite('./output/left.png',leftImage)

    img1 = cv2.imread("./output/front.png")
    img2 = cv2.imread("./final_project_data/image/front.jpg")
    img3 = cv2.imread("./output/right.png")
    img4 = cv2.imread("./final_project_data/image/right.jpg")
    img5 = cv2.imread("./output/back.png")
    img6 = cv2.imread("./final_project_data/image/back.jpg")
    img7=cv2.imread("./output/left.png")
    img8=cv2.imread("./final_project_data/image/left.jpg")
    
    orb = cv2.ORB_create(1000, 1.2)

    # Detecting keypoints of original images
    (kp1, des1) = orb.detectAndCompute(img1, None)
    (kp3, des3) = orb.detectAndCompute(img3, None)
    (kp5, des5) = orb.detectAndCompute(img5, None)
    (kp7, des7) = orb.detectAndCompute(img7, None)

    # Detecting keypoints of rotated images
    (kp2, des2) = orb.detectAndCompute(img2, None)
    (kp4, des4) = orb.detectAndCompute(img4, None)
    (kp6, des6) = orb.detectAndCompute(img6, None)
    (kp8, des8) = orb.detectAndCompute(img8, None)

   
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches1 = bf.match(des1, des2)
    matches2 = bf.match(des3, des4)
    matches3 = bf.match(des5, des6)
    matches4 = bf.match(des7, des8)

    matches1 = sorted(matches1, key=lambda val: val.distance)
    matches2 = sorted(matches2, key=lambda val: val.distance)
    matches3 = sorted(matches3, key=lambda val: val.distance)
    matches4 = sorted(matches4, key=lambda val: val.distance)

    ccfile.close()
    print ("The misalignment of front image: %.2f " %drawMatches(img1, kp1, img2, kp2, matches1))
    print ("The misalignment of right image: %.2f " % drawMatches(img3, kp3, img4, kp4, matches2))
    print ("The misalignment of back image: %.2f " %drawMatches(img5, kp5, img6, kp6, matches3))
    print ("The misalignment of left image: %.2f " %drawMatches(img7, kp7, img8, kp8, matches4))
    pass

if __name__ == '__main__':
	main()

