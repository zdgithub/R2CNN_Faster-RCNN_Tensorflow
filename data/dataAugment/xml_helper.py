# -*- coding=utf-8 -*-
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC
import os


def read_xml(xml_path):
    '''
    parse voc xml to get a list of gtbboxes
    :param xml_path:
    :return: gtnboxes list [[-1,8],...]: [x1,y1,x2,y2,x3,y3,x4,y4]
    '''
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        # name = obj.find('name').text
        box = obj.find('bndbox')
        x1 = float(box[0].text)
        y1 = float(box[1].text)
        x2 = float(box[2].text)
        y2 = float(box[3].text)
        x3 = float(box[4].text)
        y3 = float(box[5].text)
        x4 = float(box[6].text)
        y4 = float(box[7].text)
        coords.append([x1,y1,x2,y2,x3,y3,x4,y4])
    return coords


def generate_xml(img_name, coords, img_size, out_xml_path):
    '''
    输入：
        img_name：a.jpg
        coords: bboxes list
        img_size：[h,w,c]
        out_xml_path: xml save path
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('XrayData')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('Xray Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Xray Data')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:

        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode('bone')
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)

        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(round(coord[0]))))
        bndbox.appendChild(nodex1)

        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(round(coord[1]))))
        bndbox.appendChild(nodey1)

        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(round(coord[2]))))
        bndbox.appendChild(nodex2)

        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(round(coord[3]))))
        bndbox.appendChild(nodey2)

        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(round(coord[4]))))
        bndbox.appendChild(nodex3)

        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(round(coord[5]))))
        bndbox.appendChild(nodey3)

        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(round(coord[6]))))
        bndbox.appendChild(nodex4)

        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(round(coord[7]))))
        bndbox.appendChild(nodey4)


    # write doc to xml
    f = open(os.path.join(out_xml_path, img_name[:-4]+'.xml'),'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()
