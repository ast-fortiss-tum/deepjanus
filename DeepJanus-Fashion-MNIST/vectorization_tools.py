import xml.etree.ElementTree as ET
import potrace
import numpy as np
import cv2

def preprocess(npdata):
    bw = np.asarray(npdata).copy()
    # Pixel range is 0...255, 256/2 = 128
    # bw[bw < 128] = 0  # Black
    # bw[bw >= 128] = 255  # White

    # special processing for Fashion MNIST
    # hardcoding threshold works better in this case
    bw[bw < 60] = 0  # Black
    bw[bw >= 60] = 255  # White

    # bw = cv2.equalizeHist(bw)
    # tshd, bw = cv2.threshold(bw, 128, 255, cv2.THRESH_OTSU)
    # bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # tshd, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalization
    bw = bw / 255.0
    return bw


def createSVGpath(path):
    path_desc = ""
    # Iterate over path curves
    for curve in path:
        path_desc = path_desc + " M " + str(curve.start_point[0]) + "," + str(curve.start_point[1])
        for segment in curve:
            if segment.is_corner:
                path_desc = path_desc + " L " + str(segment.c[0]) + "," + str(segment.c[1]) \
                            + " L " + str(segment.end_point[0]) + "," + str(
                    segment.end_point[1])
            else:
                path_desc = path_desc + " C " + str(segment.c1[0]) + "," + str(segment.c1[1]) + " " + str(
                    segment.c2[0]) + "," + str(segment.c2[1]) + " " + str(segment.end_point[0]) + "," + str(
                    segment.end_point[1])
    return path_desc + " Z"


def create_svg_xml(desc):
    root = ET.Element("svg")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(28))
    root.set("width", str(28))
    path = ET.SubElement(root, "path")
    path.set("d", desc)
    tree = ET.ElementTree(root)
    tree = tree.getroot()
    xml_str = ET.tostring(tree, encoding='unicode', method='xml')
    return xml_str


def vectorize(image):
    array = preprocess(image)
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    desc = createSVGpath(path)
    return create_svg_xml(desc)
