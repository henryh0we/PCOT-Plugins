import cv2 as cv
import numpy as np

import pcot.config
from pcot.datum import Datum
from pcot.sources import SourceSet
from pcot.xform import XFormType, xformtype
from pcot.xforms.tabdata import TabData
from pcot.imagecube import ImageCube

# set default variables for the RIM centre and radius
RIMCX = 505
RIMCY = 307
RIMRAD = 275
MINRAD = 250
MAXRAD = 300

@xformtype
class XFormRIMDewarp(XFormType):
    """Dewarp that HRC RIMage!"""

    def __init__(self):
        # this node should appear in the maths group.
        super().__init__("Dewarp RIM", "utility", "0.0.0")
        # set input and output - they are images and are unnamed.
        self.addInputConnector("image", Datum.IMG)
        self.addOutputConnector("dewarped", Datum.IMG)

    def createTab(self, node, window):
        # there is no custom tab
        return TabData(node, window)

    def init(self, node):
        # image that will be taken from input
        node.inputImg = None
        

    def perform(self, node):
        # get the input image
        img = node.getInput(0, Datum.IMG)
        if img is not None:
            imgW,imgH,imgCh = img.shape
            # get the centre x,y and radius of the mirror
            # try Hough circles, but if no circles found, use the default values as a fall-back
            Cx = RIMCX
            Cy = RIMCY
            R = RIMRAD
            # Hough
            # find mean of all channels - construct a transform array and then use it.
            mat = np.array([1 / img.channels] * img.channels).reshape((1, img.channels))
            grey = cv.transform(img.img, mat)
            # convert to 8-bit integer from 32-bit float
            img8 = (grey * 255).astype('uint8')
            gimg = cv.medianBlur(img8,9)
            circles = cv.HoughCircles(gimg,cv.HOUGH_GRADIENT,1,500,param1=50,param2=30,minRadius=MINRAD, maxRadius=MAXRAD)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                Cx = circles[0,0][0]
                Cy = circles[0,0][1]
                R = circles[0,0][2]
            # Image input/output data
            Hs = imgH                   # source height
            Ws = imgW                   # source width
            Hd = R                      # destination height
            Wd = 2.0 * (R / 2) * np.pi  # destination width
            # prepare an empty matrix of the right size, i.e. the destination image
            map_x = np.zeros([int(Hd), int(Wd)], dtype=np.float32)
            map_y = np.zeros([int(Hd), int(Wd)], dtype=np.float32)
            # map each pixel in the destination image to a pixel in the source image
            for y in range(0, int(Hd - 1)):
                for x in range(0, int(Wd - 1)):
                    r = (float(y) / float(Hd)) * R
                    theta = (float(x)/float(Wd))* 2.0 * np.pi
                    xS = Cx + r * np.sin(theta)
                    yS = Cy + r * np.cos(theta)
                    map_x.itemset((y,x),int(xS))
                    map_y.itemset((y,x),int(yS))
            # do the unwarping 
            reprojection = cv.remap(img.img, map_x, map_y, cv.INTER_LINEAR)
            out = np.zeros(reprojection.shape)
            # move the image around so the mirror isn't split across the wrap
            x = reprojection.shape[0]
            y = reprojection.shape[1]
            left = reprojection[0:x, 0:400]     # (266, 400, 3)
            right = reprojection[0:x, 401:y]    # (266, 434, 3)
            out[0:x, 0:y-401] = right
            out[0:x, y-402:y-2] = left
            # Convert back to 32-bit float
            out = (out).astype('float32')
            # create the imagecube and set node.out for the canvas in the tab
            img = ImageCube(out, None, img.sources)
            node.out = Datum(Datum.IMG, img)
        else:
            # no image on the input, set node.out to None
            node.out = Datum.null
        # output node.out
        node.setOutput(0, node.out)