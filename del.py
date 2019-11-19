import os


if __name__ == "__main__":
    src_img_path = r'C:\Users\lenovo\Desktop\JPEGImages'
    src_xml_path = r'C:\Users\lenovo\Desktop\Annotation'

    for file in os.listdir(src_img_path):
        if file.startswith('crop'):
            img = os.path.join(src_img_path, file)
            xml = os.path.join(src_xml_path, file[:-4] + '.xml')
            os.remove(img)
            os.remove(xml)