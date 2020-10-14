# =======================================================================
# begin of file *MyImage.py*
# =======================================================================
# richard@mevis.de (c) 2005-2011

"""Modul MyImage.
"""

# history
# =======================================================================
# sept 2005: very first version
# 20-10-2005: added and corrected comments
# 11-11-2005: added comments, added member fcn stats in MyImage
# 05-12-2005: added smoothing filters, added member _getImage in MyImage
# 06-01-2006: added filter support via member fcn filter and fcn imfilter
# 15-10-2007: use pylab to show images, remove MyArray
# 23-10-2008: adapt for NumPy arrays
# 18-02-2011: converting totally to numpy, matplotlib, scipy
# 24-02-2011: some new members
# 01-04-2011: add split and merge fcns for rgb images
# 07-04-2011: save gray value images correctly
# 14-04-2011: add psnr fcn
# 15-04-2011: correct syntax error and default values in imnoise (gauss)
# 16-04-2011: use module Image in save-member
# 17-04-2011: clamp pixel values in imnoise
# 15-12-2018: print as function -> Python 3
# 14-10-2020: use imageio's imread, imwrite

# module imports
# =======================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
#import scipy.misc as msc
from PIL import Image
import scipy.ndimage.filters as flt

from math import log10

# global vars
# =======================================================================
__Version = 2.3

# class defs
# =======================================================================

# -----------------------------------------------------------------------
# begin of class *MyImage*


class MyImage(object):
    """Klasse MyImage.

    Die Klasse *MyImage* implementiert eine einfache Klasse fuer
    zweidimensionale Grauwertbilder. Bunte Bilder sind auch moeglich,
    werden aber nicht wirklich unterstuetzt.

    Unterstuetz werden:
    * das Lesen und Schreiben von Bilderdateien,
    * das Erzeugen und Kopieren von Bildern,
    * das Auslesen der Dimensionen,
    * das Lesen und Setzen von einzelnen Pixeln,
    * das Berechnen des absoluten und relativen Histogramms,
    * das Berechnen von Mittelwert und Standardabweichung,
    * das Darstellen des Bildes.     
    """

    def __init__(self, *args, **kwargs):
        """Konstruktor.

        Ein Objekt der Klasse *MyImage* kann auf drei verschiedene Arten
        erzeugt werden.

        1. img = MyImage(dateiname)
            erzeugt ein Bild aus einer Bilddatei mit Namen 'dateiname'.
        2. img = MyImage(breite,hoehe,wert)
            erzeugt ein leeres Bild mit den angegebenen Dimensionen 'breite'
            und 'hoehe'. Alle Pixel werden auf 'wert' gesetzt.
        3. img = MyImage(bild)
            erzeugt eine Kopie eines Objektes 'bild' der Klasse *MyImage*.
        """

        # self.__img is an numpy array of type float
        # self._width, self.__height are the dimensions of the image

        if len(args) == 1:
            # read an image from a file
            if isinstance(args[0], str):
                #                self.__img = msc.imread(args[0]).astype(np.float32)
                img = Image.open(args[0])
                self.__img = np.asarray(img).astype(np.float32)
            # create an image from an numpy array
            if isinstance(args[0], np.ndarray):
                self.__img = args[0].copy().astype(np.float32)
            # create an image from a MyImage object
            if isinstance(args[0], MyImage):
                self.__img = args[0].__img.copy()
        # create constant image with given dimensions
        if len(args) == 3:
            self.__img = args[2]*np.ones((args[1], args[0]), dtype=np.float32)

        # read dimensions
        # XXX: read RGB, RGBA images! depth!
        self.__height, self.__width = self.__img.shape[:2]
        self.__depth = 1
        if len(self.__img.shape) > 2:
            self.__depth = self.__img.shape[2]

        return

    def version(self):
        """Version ausgeben.
        """
        return "3.0 (c) richard 2018"

    # pixel access (read and write)
    def getPixel(self, x, y):
        """Pixel auslesen.

        Mit g = img.getPixel(x,y) wird der Pixelwert an der Stelle x,y
        ausgelesen. Liegt die Position ausserhalb des Bildes, so wird
        die Position in das Bild hineingespiegelt.
        """
        try:
            g = self.__img[y, x]
        except:
            if not(0 <= x < self.__width):
                x = (self.__width-1) - (x % self.__width)
            if not(0 <= y < self.__height):
                y = (self.__height-1) - (y % self.__height)
            g = self.__img[y, x]

        return g

    def setPixel(self, x, y, value):
        """Pixel setzen.

        Mit img.setPixel(x,y,g) wird an die Stelle 'x','y' im Bild 'img' der
        Grauwert 'g' eingesetzt. Liegt die Position ausserhalb des Bildes,
        gibt es einen Fehler.
        """
        self.__img[y, x] = value
        return

    def __getitem__(self, index):
        """Pixel auslesen.

        Mit g = img[x,y] wird der Pixelwert an der Stelle 'x','y' ausgelesen.
        Liegt die Position ausserhalb des Bildes, so wird die Position in das
        Bild hineingespiegelt.
        """
        return self.getPixel(index[0], index[1])

    def __setitem__(self, index, value):
        """Pixel setzen.

        Mit img[x,y] = g wird an die Stelle 'x','y' im Bild 'img' der Grauwert
        'g' eingesetzt. Liegt die Position ausserhalb des Bildes, gibt es einen
        Fehler.
        """
        self.setPixel(index[0], index[1], value)
        return

    # dimension access (read only)
    def getWidth(self):
        """Bildbreite lesen.

        Mit breite = img.getWidth() erhaelt man die Breite des Bildes.
        """
        return self.__width

    def getHeight(self):
        """Bildhoehe lesen.

        Mit hoehe = img.getHeight() erhalt man die Hoehe des Bildes.
        """
        return self.__height

    def getSize(self):
        return self.__width, self.__height

    def getDepth(self):
        return self.__depth

    # numpy array access
    # return the numpy array for special purposes
    def _getImage(self):
        return self.__img.copy()

    # writing
    def save(self, filename):
        """Bild speichern.

        Ueber img.save(dateiname) wird das Bild in 'img' in die Datei
        'dateiname' geschrieben. Die Endung von 'dateiname' bestimmt das
        Bildformat.
        """
        if len(self.__img.shape) == 2:  # gray
            #            cmap = plt.cm.gray
            #            img = msc.toimage(self.__img.astype('uint8'))
            img = Image.fromarray(self.__img.astype('uint8'))
            img.save(filename)
#            plt.imsave(filename, self.__img.astype('uint8'), cmap=cmap, vmin=0, vmax=255)
#            msc.imsave(filename, self.__img)  # scales the image!!!
        else:
            plt.imsave(filename, self.__img/255)
        return

    # showing
    def show(self):
        """Bild anzeigen.

        Der Befehl img.show() zeigt das Bild in einem Fenster an.
        """

        if len(self.__img.shape) == 2:  # gray
            cmap = plt.cm.gray
            plt.imshow(self.__img, cmap=cmap, interpolation='nearest',
                       vmin=0, vmax=255)
        else:  # color
            plt.imshow(self.__img/255, interpolation='nearest',
                       vmin=0, vmax=1)

        plt.axis('off')
        plt.show()
        return

    # point operator
    # Use numpy fcns in the definition of f!
    def point(self, f):
        return MyImage(f(self.__img))

    # histogram fcns
    def absHistogram(self):
        """Absolutes Histogramm berechnen.

        Mit H = img.abs_histogram() erhaelt man das absolute Histogramm
        des Bildes 'img' in der Liste 'H' zurueck.
        """
        return np.histogram(self.__img, 256, (0, 256))[0]

    def relHistogram(self):
        """Relatives Histogramm berechnen.

        Mit h = img.rel_histogram() erhaelt man das relative Histogramm
        des Bildes 'img' in der Liste 'h' zurueck.
        """
        return np.histogram(self.__img, 256, (0, 256), True)[0]

    def cumHistogram(self):
        H = self.relHistogram()
        C = H.copy()
        for i in range(1, len(C)):
            C[i] = C[i-1]+C[i]
        return C

    def stats(self):
        """Mittelwert und Streuung berechnen.

        Mit (eta,s) = img.stats() erhaelt man in 'eta' den mittleren Grauwert
        des Bildes 'img' und in 's' die Streuung.
        """
        return self.__img.mean(), self.__img.std()

    def minmax(self):
        return self.__img.min(), self.__img.max()

# end of class *MyImage*
# -----------------------------------------------------------------------

# fcn defs
# =======================================================================

# -----------------------------------------------------------------------
# noise fcn coded along imnoise.m from Matlab


def imnoise(img, name, *args):
    """Verrauscht eine gegebenes Bild.

    1. out = imnoise(img,'gauss',mu,nu)
      verrauscht nach Gauss mit Mittelwert *mu* und Varianz *nu*.
    2. out = imnoise(img,'salt',dense)
      verrauscht mit Salt und Pepper mit Dichte *dense*.
    3. out = imnoise(img,'speckle',intense)
      verrauscht mit Speckle und Intensitaet *intense*.
    """
    out = MyImage(img)
    if name[:2] == 'ga':
        if len(args) == 2:
            mu, var = args
        elif len(args) == 1:
            mu, var = args[0], 0.5*255
        else:
            mu, var = 0, 0.5*255
        sigma = var**0.5

        for y in range(out.getHeight()):
            for x in range(out.getWidth()):
                r = random.gauss(mu, sigma)
                g = out[x, y] = out[x, y] + r
                if g < 0.:
                    out[x, y] = 0.
                if g > 255.:
                    out[x, y] = 255.

    elif name[:2] == 'sa':
        if len(args) == 1:
            dense = args[0]
        else:
            dense = 0.05
        denseh = 0.5*dense

        for y in range(out.getHeight()):
            for x in range(out.getWidth()):
                r = random.random()
                if r < denseh:
                    out[x, y] = 0
                elif 1.-denseh <= r:
                    out[x, y] = 255
                else:
                    pass

    elif name[:2] == 'sp':
        if len(args) == 1:
            intense = args[0]
        else:
            intense = 0.04

        factor = (12*intense)**0.5
        for y in range(out.getHeight()):
            for x in range(out.getWidth()):
                r = random.random()-0.5
                g = out[x, y] = out[x, y] * (1 + factor*r)
                if g < 0.:
                    out[x, y] = 0.
                if g > 255.:
                    out[x, y] = 255.

    else:
        print("unknown noise type")

    return out

#
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------


def split(img):
    """Return color channels of an RGB image.
    """

    if img.getDepth() == 3:
        arr = img._getImage()
        r = MyImage(arr[:, :, 0])
        g = MyImage(arr[:, :, 1])
        b = MyImage(arr[:, :, 2])
    else:
        r = img.copy()
        g = img.copy()
        b = img.copy()

    return r, g, b


def merge(r, g, b):
    """Combine single RGB channels into a color image.
    """

    R, G, B = r._getImage(), g._getImage(), b._getImage()

    h, w = r.getHeight(), r.getWidth()
    RGB = np.zeros((h, w, 3), np.float32)
    RGB[:, :, 0] = R
    RGB[:, :, 1] = G
    RGB[:, :, 2] = B

    return MyImage(RGB)


def MedianFilter(img, size=3):
    """Median gefiltertest Bild berechnen.

    out = MedianFilter(img,size)

    Darin bezeichnet *size* die Groesse der Umgebung, die fuer den Median
    verwendet wird
    """
    arr_img = img._getImage()
    arr_out = flt.median_filter(arr_img, size)
    return MyImage(arr_out)

# filter fcn ala matlab.


def imfilter(img, mask, boundary='symmetric'):
    """Filtert ein Bild mit der gegebenen Maske.

    boundary = 'symmetric' (default), 'replicate', 'circular', value
    """

    # create numpy array from image
    if isinstance(img, MyImage):
        arr_img = img._getImage()
    else:
        arr_img = np.array(img, dtype=np.float)

    if len(arr_img.shape) > 2:
        arr_img = arr_img[:, :, 0]

    if boundary == 'symmetric':
        mode, cval = 'reflect', 0.0
    elif boundary == 'replicate':
        mode, cval = 'nearest', 0.0
    elif boundary == 'circular':
        mode, cval = 'wrap', 0.0
    else:
        mode, cval = 'constant', boundary

    arr_out = flt.correlate(arr_img, mask, mode=mode, cval=cval)

    return MyImage(arr_out)

#
# -----------------------------------------------------------------------


def psnr(img1, img2):
    w, h = img1.getWidth(), img1.getHeight()
    N = w*h
    Min, Max = img2.minmax()
    s = 0.
    for x in range(w):
        for y in range(h):
            s += (img1[x, y]-img2[x, y])**2
    return 10.*log10((Max**2*N)/s)


# script entry
# =======================================================================
if __name__ == '__main__':
    pass

# =======================================================================
# end of file *MyImage.py*
# =======================================================================
