import xml.etree.ElementTree as ElementTree


def obtain_box(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = {
            'class': label,
            'y_min': float(xml_box.find('ymin').text),
            'x_min': float(xml_box.find('xmin').text),
            'y_max': float(xml_box.find('ymax').text),
            'x_max': float(xml_box.find('xmax').text)
        }
        boxes.append(bbox)

    return boxes

def extract_boxes(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:

        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = {
            'class': label,
            'y_min': round(float(xml_box.find('ymin').text)),
            'x_min': round(float(xml_box.find('xmin').text)),
            'y_max': round(float(xml_box.find('ymax').text)),
            'x_max': round(float(xml_box.find('xmax').text))
        }

        boxes.append(bbox)
    return boxes, width, height

def count_classes(classes):
    return len(classes)

