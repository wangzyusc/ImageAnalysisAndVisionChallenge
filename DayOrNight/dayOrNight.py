import sys

img = list()

for line in sys.stdin:
    imgline = list()
    dots = line.split()
    for pixel in dots:
        rgb = pixel.split(',')
        imgline.append((int(rgb[0]) + int(rgb[1]) + int(rgb[2])) / 3)
    img.append(imgline)

total = 0
for line in img:
    total = total + sum(line)

rows = len(img)
cols = len(img[0])
avg = float(total) / (rows * cols)

if avg >= 80:
    print 'day'
else:
    print 'night'
