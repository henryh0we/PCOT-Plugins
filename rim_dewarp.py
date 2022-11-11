import cv2 as cv
import numpy as np

from pcot.sources import SourceSet
from pcot.xform import XFormType, xformtype, Datum
from pcot.xforms.tabdata import TabData
from pcot.imagecube import ImageCube

import pcot.config

@xformtype
class XFormEdgeDetect(XFormType):
    """Dewarp that HRC RIMage!"""

    def __init__(self):
        # this node should appear in the maths group.
        super().__init__("Dewarp RIM", "utility", "0.0.0")
        # set input and output - they are images and are unnamed.
        self.addInputConnector("", Datum.IMG)
        self.addOutputConnector("", Datum.IMG)

    def createTab(self, n, w):
        # there is no custom tab, we just use an data canvas. This expects "node.out" to be set to
        # either None or an imagecube.
        return TabData(n, w)

    def init(self, n):
        # No initialisation required.
        pass

    def perform(self, node):
        # get the input image
        img = node.getInput(0, Datum.IMG)
        if img is not None:
            imgW,imgH,imgCh = img.shape
            # centre x,y and radius of the mirror  
            Cx = 426
            Cy = 447
            R = 275
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