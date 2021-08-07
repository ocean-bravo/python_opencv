import cv2 as cv

windows = {}

im1 = 0
im2 = 0
im3 = 0
im4 = 0
im5 = 0

gaussData = {"ksize": 1, "sigmaX": 0, "sigmaY": 0}
cannyData = {"threshold1": 70, "threshold2": 180}
cropData = {"x1": 10, "x2": 100, "y1": 10, "y2": 100}


def universalCallback(value, winname, propertyname):
    windows[winname][propertyname] = value


def gaussBlur(winname, inImg):
    data = windows[winname]
    ksize = data["ksize"]

    # ksize должно быть нечетным
    if ksize % 2 == 0:
        ksize += 1

    sigmaX = float(data["sigmaX"])/100
    sigmaY = float(data["sigmaY"])/100

    im = cv.GaussianBlur(inImg, (ksize, ksize), sigmaX, sigmaY)
    cv.imshow(winname, im)
    return im


def canny(winname, inImg):
    data = windows[winname]
    threshold1 = data["threshold1"]
    threshold2 = data["threshold2"]

    im = cv.Canny(inImg, threshold1, threshold2)
    cv.imshow(winname, im)
    return im


def crop(winname, inImg):
    data = windows[winname]
    x1 = data["x1"]
    x2 = data["x2"]
    y1 = data["y1"]
    y2 = data["y2"]

    if x1 >= x2:
        x2 = x1 + 1

    if y1 >= y2:
        y2 = y1 + 1

    im = inImg[x1:x2, y1:y2]
    cv.imshow(winname, im)
    return im


def init():
    global im1
    im1 = cv.imread("./01-01385/25.jpeg")
    cv.imshow("stage1", im1)
    cv.moveWindow("stage1", 0, 0)

    global im2
    im2 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    cv.imshow("stage2", im2)
    cv.moveWindow("stage2", 400, 0)

    win3 = "gauss"
    cv.namedWindow(win3)
    cv.moveWindow(win3, 400, 0)
    windows[win3] = gaussData
    cv.createTrackbar("sigmaX", win3, 0, 100, lambda x: universalCallback(x, win3, "sigmaX"))
    cv.createTrackbar("ksize", win3, 1, 500, lambda x: universalCallback(x, win3, "ksize"))
    cv.createTrackbar("sigmaY", win3, 0, 100, lambda x: universalCallback(x, win3, "sigmaY"))

    global win4
    win4 = "canny"
    cv.namedWindow(win4)
    cv.moveWindow(win4, 800, 0)
    windows[win4] = cannyData
    cv.createTrackbar("threshold1", win4, 0, 255, lambda x: universalCallback(x, win4, "threshold1"))
    cv.createTrackbar("threshold2", win4, 0, 255, lambda x: universalCallback(x, win4, "threshold2"))
    cv.setTrackbarPos("threshold1", win4, 70)
    cv.setTrackbarPos("threshold2", win4, 180)

    global win5
    win5 = "crop"
    cv.namedWindow(win5)
    cv.moveWindow(win5, 1200, 0)
    windows[win5] = cropData
    height, width, _ = im1.shape
    cv.createTrackbar("x1", win5, 0, height, lambda x: universalCallback(x, win5, "x1"))
    cv.createTrackbar("x2", win5, 0, height, lambda x: universalCallback(x, win5, "x2"))
    cv.createTrackbar("y1", win5, 0, width, lambda x: universalCallback(x, win5, "y1"))
    cv.createTrackbar("y2", win5, 0, width, lambda x: universalCallback(x, win5, "y2"))

    global im5
    cv.createButton("save crop", lambda x, y: cv.imwrite("template.jpeg", im5))


def loop():
    while True:
        global im3
        im3 = gaussBlur("gauss", im2)

        global im4
        im4 = canny("canny", im3)

        global im5
        im5 = crop("crop", im4)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    init()
    loop()
    cv.destroyAllWindows()
