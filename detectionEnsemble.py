import os
from imutils import paths
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats
from xml.dom import minidom


# The paramater of the function is a path that contains the predictions of the
def nonMaximumSupression(detections_path):
    listdirmodels = [ p for p in os.listdir(detections_path) if "detection" in p]
    annotationsFiles = list(paths.list_files(os.path.join(listdirmodels[0]), validExts=(".xml")))
    for an in annotationsFiles:
        boxes = []
        classesBoxes = []
        fileName = an.split("/")[-1]
        # boxes += extractBoxes(an)
        for dir in listdirmodels:
            if os.path.isdir(dir):
                ImageBoxes, classesB = extractBoxes(os.path.join(dir,fileName))
                if len(ImageBoxes)!=0:
                    boxes = boxes + ImageBoxes
                    classesBoxes = classesBoxes + classesB

        # boxes=[extractBoxes(os.path.join(dir,fileName)) for dir in listdirmodels if os.path.isdir(dir)]
        boxes = np.array(boxes)
        classesBoxes = np.array(classesBoxes)
        if(len(boxes)!=0):
            boxes, modes = non_max_suppression_fast(boxes,classesBoxes,0.45)

        xml =generateXML(an, boxes, modes, "detectiEnsemble")
        file = open(os.path.join(".","detectionEnsemble",fileName),'w')
        file.write(xml)


def extractBoxes(annotation_path):
    boxes = []
    classes = []
    doc = ET.parse(annotation_path)
    doc = doc.getroot()
    objects = doc.findall("object")
    for o in objects:
        box = []
        bndBox = o.find('bndbox')
        name = o.find('name').text
        confidence = o.find('confidence').text
        box.append(int(bndBox.find('xmin').text))
        box.append(int(bndBox.find('ymin').text))
        box.append(int(bndBox.find('xmax').text))
        box.append(int(bndBox.find('ymax').text))
        classes.append(name)
        box.append(float(confidence))
        boxes.append(box)
    return boxes,classes

# Malisiewicz et al.
def non_max_suppression_fast(boxes,classesBoxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    modes = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        # i es el indice del elemento que se mantiene
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxDeleted =  np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        auxidxs = np.append(idxDeleted, i)
        x = []

        for j in auxidxs:
            x.append(classesBoxes[j])

        mode = stats.mode(x)
        idxs = np.delete(idxs,idxDeleted)
        np.delete(classesBoxes, auxidxs)
        modes.append(mode[0][0])
    # return only the bounding boxes that were picked using the
    # integer data type
    boxes[pick].astype("int"), modes
    return boxes[pick].astype("int") , modes


def generateXML(annotationFile, boxes,categories,outputPath):
    doc = ET.parse(annotationFile)
    doc = doc.getroot()
    # doc = doc.find('annotation')
    filename = doc.find('filename').text
    path = doc.find('path').text
    size = doc.find('size')
    w = size.find('width').text
    h = size.find('height').text

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(3)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)
    for box, category in zip(boxes, categories):
        confidence=1.0
        if(len(box)==2):
            (xmin,ymin,xmax,ymax, con) = box
        else:
            (xmin,ymin,xmax,ymax,con) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childConfidence = ET.SubElement(childObject, 'confidence')
        childConfidence.text = str(confidence)
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(xmin)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(ymin)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(xmax)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(ymax)
    return prettify(top)



def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")







