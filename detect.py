import cv2 as cv

windows = {}

im1 = 0
im2 = 0
im3 = 0
im4 = 0

template = 0

gaussData = {"ksize": 1, "sigmaX": 0, "sigmaY": 0}
cannyData = {"threshold1": 70, "threshold2": 180}


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


def init():
    global im1
    im1 = cv.imread("./01-01385/1.jpeg")
    cv.imshow("stage1", im1)
    cv.moveWindow("stage1", 0, 0)

    global template
    template = cv.imread("./bad.jpeg", cv.IMREAD_GRAYSCALE) # обязательно серым должно быть
    cv.imshow("template", template)

    global im2
    im2 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    cv.imshow("stage2", im2)
    cv.moveWindow("stage2", 100, 0)

    win3 = "gauss"
    cv.namedWindow(win3)
    cv.moveWindow(win3, 200, 0)
    windows[win3] = gaussData
    cv.createTrackbar("sigmaX", win3, 0, 100, lambda x: universalCallback(x, win3, "sigmaX"))
    cv.createTrackbar("ksize", win3, 1, 500, lambda x: universalCallback(x, win3, "ksize"))
    cv.createTrackbar("sigmaY", win3, 0, 100, lambda x: universalCallback(x, win3, "sigmaY"))

    global win4
    win4 = "canny"
    cv.namedWindow(win4)
    cv.moveWindow(win4, 300, 0)
    windows[win4] = cannyData
    cv.createTrackbar("threshold1", win4, 0, 255, lambda x: universalCallback(x, win4, "threshold1"))
    cv.createTrackbar("threshold2", win4, 0, 255, lambda x: universalCallback(x, win4, "threshold2"))
    cv.setTrackbarPos("threshold1", win4, 70)
    cv.setTrackbarPos("threshold2", win4, 180)


def loop():
    while True:
        global im3
        im3 = gaussBlur("gauss", im2)

        global im4
        im4 = canny("canny", im3)

        #methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
        methods = [cv.TM_CCOEFF]
        for method in methods:
            img = im4.copy()
            result = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            cv.normalize(result,result,1,0,32);
            cv.imshow("result", result)

            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                location = min_loc
            else:
                location = max_loc

            h, w = template.shape
            bottom_right = (location[0] + w, location[1] + h)
            cv.rectangle(img, location, bottom_right, (255, 0, 0), 2)
            cv.imshow('Match', img)
            cv.moveWindow('Match', 800, 0)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    init()
    loop()
    cv.destroyAllWindows()
