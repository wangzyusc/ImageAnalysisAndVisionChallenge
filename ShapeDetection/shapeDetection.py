import numpy as np, cv2, math, sys

"""
    This method find the best threshold for binarization of input image.
    I.e. for a grayscale image, pixel >= threshold would be 255, or 0 otherwise. 
    Assume that input is a two-dimensional list.
"""
def binaryThreshold(img):
    hist = [0 for i in range(256)]
    rows = len(img)
    cols = len(img[0])
    #print 'binarythreshold: rows: ' + str(rows) + ' cols: ' + str(cols)
    for i in range(rows):
        for j in range(cols):
            #print 'img[' + str(i) + '][' + str(j) + '] = ' + str(img[i][j])
            hist[img[i][j]] = hist[img[i][j]] + 1
    totalNum = 0
    for num in hist:
        totalNum = totalNum + num
    min = 0
    max = 255
    while hist[min] == 0:
        min = min + 1
    while hist[max] == 0:
        max = max - 1
    if min == max:
        #print 'box'
        return 0
    i = min
    sum = 0
    while i <= max:
        sum = sum + i * hist[i]
        i = i + 1
    ufg = float(sum) / float(totalNum)
    ubg = 0
    Nfg = totalNum
    Nbg = 0
    maxSigma = float(0)
    result = 0
    if debug:
        print 'min = ' + str(min) + ' max = ' + str(max)
    for T in range(min+1, max+1):
        ubg = float(float(ubg) * Nbg + hist[T-1] * (T-1))
        ufg = float(float(ufg) * Nfg - hist[T-1] * (T-1))
        Nbg = Nbg + hist[T-1]
        Nfg = Nfg - hist[T-1]
        #print 'hist[' + str(T) + '] = ' + str(hist[T-1]) + ' Nbg = ' + str(Nbg) + ' Nfg = ' + str(Nfg)
        ubg = float(ubg) / float(Nbg)
        ufg = float(ufg) / float(Nfg)
        sigma = (float(Nbg * Nbg) / float(totalNum * totalNum)) * float(ufg - ubg) * float(ufg - ubg)
        #print 'Nbg = ' + str(Nbg) + ' Nfg = ' + str(Nfg) + ' ubg = ' + str(ubg) + ' ufg = ' + str(ufg) + ' sigma(' + str(T) + ') = ' + str(sigma)
        if sigma > maxSigma:
            maxSigma = sigma
            result = T
    return result

"""
    This method is to do the binarization of an input image.
    It utilizes the previous method to find the threshold.
    Assumes that the input is a two-dimensional list.
"""
def binarization(img):
    res = list()
    threshold = binaryThreshold(img)
    if threshold != 0:
        relaxation = 5
        for row in img:
            line = list()
            for pixel in row:
                if pixel >= threshold - relaxation:
                    line.append(255)
                else:
                    line.append(0)
            res.append(line)
    return res, threshold

"""
    Helper method to determine whether a coordinate (x,y) is in
    the bound of an image with width w and height h. If the coordinate
    is in bound, return True.
"""
def inbound(x, y, w, h):
    return (x >= 0) and (x < w) and (y >= 0) and (y < h)

"""
    This method implements the erosion operation in binary image
    morphology. It uses a circle kernel to do the erosion, whose
    radius is the 2nd input parameter.
"""
def erode(src, radius):
    dst = list()
    rows = len(src)
    cols = len(src[0])
    musk = list()
    for y in range(-radius, radius+1):
        muskline = list()
        for x in range(-radius, radius+1):
            if(x*x + y*y <= radius*radius):
                muskline.append(True)
            else:
                muskline.append(False)
        musk.append(muskline)
    for i in range(rows):
        dstline = list()
        for j in range(cols):
            val = 255
            for x in range(-radius, radius+1):
                if val == 0:
                    break
                for y in range(-radius, radius+1):
                    if inbound(j+x,i+y,cols,rows) and (musk[y][x] == True):
                        if(src[i+y][j+x] == 0):
                            val = 0
                            break
            dstline.append(val)
        dst.append(dstline)
    return dst

"""
    Similar to erode method, but this one is implementation of dilation.
"""
def dilate(src, radius):
    dst = list()
    rows = len(src)
    cols = len(src[0])
    musk = list()
    for y in range(-radius, radius+1):
        muskline = list()
        for x in range(-radius, radius+1):
            if(x*x + y*y <= radius*radius):
                muskline.append(True)
            else:
                muskline.append(False)
        musk.append(muskline)
    for i in range(rows):
        dstline = list()
        for j in range(cols):
            val = 0
            for x in range(-radius, radius+1):
                if val == 255:
                    break
                for y in range(-radius, radius+1):
                    if (inbound(j+x,i+y,cols,rows) == True) and (musk[y][x] == True):
                        if(src[i+y][j+x] == 255):
                            val = 255
                            break
            dstline.append(val)
        dst.append(dstline)
    return dst

"""
    Implemented open operation in binary image morphology. First do dilation
    to the input image and then do erosion to the dilation result.
    Can effectively reduce small noises in the image.
"""
def denoise(src, radius):
    #return dilate(erode(src,radius),radius)
    return erode(dilate(src,radius),radius)

"""
    Implemented simple downsample operation for image.
"""
def downsample(img, rate):
    rows = len(img)
    cols = len(img[0])
    drows = rows / rate
    dcols = cols / rate
    res = list()
    for i in range(drows):
        line = list()
        for j in range(dcols):
            sum = 0
            for ii in range(rate):
                for jj in range(rate):
                    sum = sum + img[i * rate + ii][j * rate + jj]
            line.append(sum / (rate * rate))
        res.append(line)
    return res

"""
    Implemented Sobel operator.
"""
def Sobel(src):
    Ix = list()
    Iy = list()
    rows = len(src)
    cols = len(src[0])
    for i in range(1, rows - 1):
        xline = list()
        yline = list()
        for j in range(1, cols - 1):
            xline.append(src[i-1][j+1] + 2 * src[i][j+1] + src[i+1][j+1] - src[i-1][j-1] - 2 * src[i][j-1] - src[i+1][j-1])
            yline.append(src[i-1][j-1] + 2 * src[i-1][j] + src[i-1][j+1] - src[i+1][j-1] - 2 * src[i+1][j] - src[i+1][j+1])
        Ix.append(xline)
        Iy.append(yline)
    return Ix, Iy

"""
    Compute the magnitude and direction(in radius) of edges from Ix and Iy,
    which is calculated by Sobel operator.
"""
def getEdge(Ix, Iy):
    magnitude = list()
    direction = list()
    rows = len(Ix)
    cols = len(Ix[0])
    for i in range(rows):
        magline = list()
        dirline = list()
        for j in range(cols):
            magline.append(math.sqrt(Ix[i][j]*Ix[i][j] + Iy[i][j]*Iy[i][j]))
            if Ix[i][j] == 0:
                if Iy[i][j] == 0:
                    dirline.append(0)
                if Iy[i][j] > 0:
                    dirline.append(math.pi / 2)
                if Iy[i][j] < 0:
                    dirline.append(-math.pi / 2)
            else:
                dirline.append(math.atan(Iy[i][j] / Ix[i][j]))
        magnitude.append(magline)
        direction.append(dirline)
    return magnitude, direction

"""
    Given a direction of edge, classify the angel to 8 categories.
"""
def angelClassify(angel):
    #divide the -pi/2 to pi/2 range by 8, so have 8 regions
    #bound is: [[7/16*pi, -7/16*pi],[-7/16*pi, -5/16*pi],[-5/16*pi, -3/16*pi],[-3/16*pi, -1/16*pi],[-1/16*pi, 1/16*pi],[1/16*pi, 3/16*pi],[1/16*pi, 3/16*pi],[3/16*pi, 5/16*pi],[5/16*pi, 7/16*pi]]
    fraction = angel / math.pi;
    if fraction < float(-7) / 16:
        return 0
    elif fraction < float(-5) / 16:
        return 1
    elif fraction < float(-3) / 16:
        return 2
    elif fraction < float(-1) / 16:
        return 3
    elif fraction < float(1) / 16:
        return 4
    elif fraction < float(3) / 16:
        return 5
    elif fraction < float(5) / 16:
        return 6
    elif fraction < float(7) / 16:
        return 7
    else:
        return 0

"""
    Classify the image to various shapes according to the 
    magnitude and direction of edges.
"""
def shapeClassify(magnitude, direction, mag_threshold):
    rows = len(magnitude)
    cols = len(magnitude[0])
    maxrange = math.pi / 2
    minrange = -math.pi / 2
    count = [0 for i in range(8)]
    for i in range(rows):
        for j in range(cols):
            if magnitude[i][j] >= mag_threshold:
                pos = angelClassify(direction[i][j])
                count[pos] = count[pos] + 1
    totalCount = sum(count)
    #freq = count / float(totalCount)
    freq = list()
    for num in count:
        if totalCount != 0:
            freq.append(float(num) / totalCount)
            #print num
    edgeNum = 0
    triangleNum = 0
    for fnum in freq:
        if debug:
            print fnum
        if fnum > 0.05:
            edgeNum = edgeNum + 1
        if fnum > 0.17:
            triangleNum = triangleNum + 1
        #if freq[i] > 0.1:
            #edgeNum = edgeNum + 1
    if debug:
        print 'edgeNum = ' + str(edgeNum)
        print 'triangleNum = ' + str(triangleNum)
    if edgeNum == 2:
        return 'box'
    elif edgeNum >= 6:
        return 'circleORellipse'
    elif triangleNum == 3:
        return 'triangle'

"""
    Classify circles and ellipes.
    Assume the input src is denoised matrix represented as list.
"""
def isCircle(src):
    rows = len(src)
    cols = len(src[0])
    bg = 255
    fg = 0
    if src[0][0] == 0:
        bg = 0
        fg = 255
    ymin = rows - 1
    ymax = 0
    xmin = cols - 1
    xmax = 0
    for y in range(rows):
        for x in range(cols):
            if src[y][x] == fg:
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
    ratio = float(ymax - ymin) / float(xmax - xmin)
    if (ratio > 0.9) and (ratio < 1.1):
        return 'circle'
    else:
        return 'ellipse'

"""
    Below is main method.
"""
debug = False
if (len(sys.argv) == 2) and sys.argv[1] == 'debug':
    debug = True

nums = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']
path = '/Users/zhiyuanwang/Documents/Workspace/PythonProjects/ShapeDetection/testcases/input/input'
suffix = '.txt'

files = list()
mats = list()

if debug:
    cv2.namedWindow('original')
    cv2.namedWindow('binary')
    #cv2.namedWindow('downsample')
    cv2.namedWindow('denoised')
    cv2.namedWindow('sobelx')
    cv2.namedWindow('sobely')
    cv2.namedWindow('edge')
    cv2.namedWindow('edgeDirection')

for num in nums:
#for num in range(12,13):
    filepath = path + str(num) + suffix
    if debug:
        print filepath
    img_data = list()
    hist = [0 for i in range(256)]
    with open(filepath) as f:
        for line in f:
            imgline = list()
            for pixel in line.split():
                rgb = pixel.split(',')
                grayscale = (int(rgb[0])+int(rgb[1])+int(rgb[2]))/3
                imgline.append(grayscale)
                hist[grayscale] = hist[grayscale] + 1
            img_data.append(imgline)
    files.append(img_data)
    #txt data reading finished, do gaussian blur and binarization next
    #use downsampling instead of gaussian blur:
    rows = len(img_data)
    cols = len(img_data[0])
    if debug:
        print 'rows: ' + str(rows) + ' cols: ' + str(cols)
    sample_rate = 5
    #downsampled_data = downsample(img_data, sample_rate)
    binary_data, threshold = binarization(img_data)
    if debug:
        print 'threshold: ' + str(threshold)
    if threshold != 0:
        radius = 2
        denoised_data = denoise(binary_data, radius)
        Ix, Iy = Sobel(denoised_data)
        #Use Ix and Iy to calculate the edge angel, then cluster the angels to judge the shape
        magnitude, direction = getEdge(Ix, Iy)
        low = min(min(direction))
        high = max(max(direction))
        if debug:
            print 'direction min = ' + str(low) + ' max = ' + str(high)
        mag_threshold = 200
        shape = shapeClassify(magnitude, direction, mag_threshold)
        if shape == 'circleORellipse':
            shape = isCircle(denoised_data)
    else:
        shape = 'box'
    print shape
    if debug:
        img = np.array(img_data) / 255.0
        binary = np.array(binary_data) / 255.0
        denoised = np.array(denoised_data) / 255.0
        #downsampled = np.array(downsampled_data) / 255.0
        sobelx = np.array(Ix) / 255.0
        sobely = np.array(Iy) / 255.0
        edge = np.array(magnitude) / 255.0
        edgeDirection = np.array(direction) * 2.0 / math.pi + 0.5
        cv2.imshow('original',img)
        cv2.imshow('binary', binary)
        #cv2.imshow('downsample', downsampled)
        cv2.imshow('denoised', denoised)
        cv2.imshow('sobelx',sobelx)
        cv2.imshow('sobely',sobely)
        cv2.imshow('edge', edge)
        cv2.imshow('edgeDirection',edgeDirection)
        cv2.waitKey(0)

if debug:
    cv2.destroyAllWindows()
