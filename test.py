import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(img, name="", t=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(t)
    cv2.destroyAllWindows()


def onmouse(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        params[y, x] = 1
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        params[y, x] = 0


if __name__ == "__main__":
    a = np.arange(16).reshape((4,4, 1)).astype(np.uint8)
    b = np.tile(a, (1,1,3))













    path = "./20210830113621.jpg"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # show(gray)
    # # _, otsu = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    # otsu = np.where((gray>150) & (gray<155), 255, 0)
    # otsu = otsu.astype(np.uint8)
    # show(otsu)
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()

    rect = (105, 170, 780, 490)
    mask = np.ones_like(gray, np.uint8) * 2

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("mask", onmouse, mask)

    while True:
        cv2.imshow("mask", img)
        if cv2.waitKey(30) & 0xff == 27:
            break

    cv2.destroyAllWindows()

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

    # mask_show = mask * 80
    mask_show = np.where((mask == 1) | (mask == 3), 1, 0)
    mask_show = mask_show.astype(np.uint8)

    img = cv2.bitwise_and(img, img, mask=mask_show)
    show(img)

    gray = cv2.GaussianBlur(gray, (3,3), 1)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(dx, dy)
    grad *= mask_show
    grad = grad.astype(np.uint8)
    ker = np.ones((3, 3))
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, ker)
    show(grad)

    _, test = cv2.threshold(grad, 150, 255, cv2.THRESH_BINARY_INV)
    show(test)


    pass
