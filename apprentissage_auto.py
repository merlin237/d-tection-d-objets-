#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess

# Chemin vers votre fichier tar
tar_file = 'VOCtrainval_11-May-2012.tar'

try:
    # Exécuter la commande tar pour extraire les fichiers
    subprocess.run(['tar', '-xvf', tar_file], check=True)
    print(f"Extraction de {tar_file} réussie.")
except subprocess.CalledProcessError as e:
    print(f"Erreur lors de l'extraction : {e}")


# In[48]:


import os
import cv2
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET


# In[49]:


dataset_path = "C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/"
images_path = os.path.join(dataset_path, "JPEGImages")
annotations_path = os.path.join(dataset_path, "Annotations")


# In[50]:


# Sélectionne une image aléatoire
image_name = os.listdir(images_path)[0]
image_path = os.path.join(images_path, image_name)

# Charge l'image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chemin du fichier d'annotation correspondant
annotation_name = os.path.splitext(image_name)[0] + ".xml"
annotation_path = os.path.join(annotations_path, annotation_name)

# Parse l'annotation XML
tree = ET.parse(annotation_path)
root = tree.getroot()

# Affiche les informations de l'annotation
for obj in root.findall('object'):
    label = obj.find('name').text
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    # Dessine le rectangle sur l'image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    plt.text(xmin, ymin - 10, label, color='red', fontsize=12)

# Affiche l'image avec les annotations
plt.imshow(image)
plt.axis('off')
plt.show()


# In[51]:


import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(annotations_dir, output_dir, class_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Créer un dictionnaire de mappage des noms de classes aux indices
    class_to_index = {name: index for index, name in enumerate(class_names)}

    for file_name in os.listdir(annotations_dir):
        if file_name.endswith('.xml'):
            xml_file = os.path.join(annotations_dir, file_name)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)

            with open(os.path.join(output_dir, file_name.replace('.xml', '.txt')), 'w') as f:
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label not in class_to_index:
                        # Ignorer les classes non définies dans class_names
                        continue
                    
                    class_index = class_to_index[label]
                    bbox = obj.find('bndbox')
                    
                    # Convertir les coordonnées en flottant
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # Normalisation des coordonnées
                    x_center = (xmin + xmax) / 2.0 / image_width
                    y_center = (ymin + ymax) / 2.0 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    # YOLO format: <object-class> <x_center> <y_center> <width> <height>
                    f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

# Liste des classes du dataset Pascal VOC 2012
class_names = [
    'person', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Exécute la conversion
path_to_your_dataset = 'C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/'
annotations_dir = os.path.join(path_to_your_dataset, "Annotations")
output_dir = os.path.join(path_to_your_dataset, "labels")

convert_voc_to_yolo(annotations_dir, output_dir, class_names)


# In[52]:


import os
import xml.etree.ElementTree as ET

# Dossiers
input_dir = "C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/Annotations"  # Dossier des fichiers XML
output_dir = "C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/labels"      # Dossier de sortie des fichiers .txt
image_dir = "C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/JPEGImages"    # Dossier des images

# Liste des classes
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    output_file = os.path.join(output_dir, os.path.basename(xml_file).replace(".xml", ".txt"))
    with open(output_file, "w") as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

# Parcourir tous les fichiers XML et les convertir
for xml_file in os.listdir(input_dir):
    if xml_file.endswith(".xml"):
        convert_annotation(os.path.join(input_dir, xml_file))

print("Conversion terminée.")


# In[41]:


import os
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = r'C:\Program Files\Git\bin\git.exe'


# In[53]:


get_ipython().system('python C:\\Users/USER/Documents/tppython/yolov5-master/train.py --img 640 --batch 16 --epochs 50 --data C:/Users/USER/Documents/tppython/voc2012.yaml --weights yolov5s.pt --cache')


# In[47]:


#  Lancement de l'entraînement
get_ipython().system('python C:/Users/USER/Documents/tppython/yolov5-master/train.py --weights yolov5s.pt --data C:/Users/USER/Documents/tppython/voc2012.yaml --epochs 50 --batch-size 16 --imgsz 640')


# In[ ]:





# In[46]:


import matplotlib.pyplot as plt

# Charger les résultats de l'entraînement
results = "path_to_your_training_folder/results.txt"

# Lire les résultats
data = open(results, 'r').readlines()

# Extraire les données souhaitées (par exemple, mAP à 0.5)
epochs = []
map50 = []

for line in data:
    if line.startswith("Epoch"):
        continue
    line = line.strip().split()
    epochs.append(int(line[0]))
    map50.append(float(line[7]))  # mAP@0.5

# Tracer les résultats
plt.plot(epochs, map50, label='mAP@0.5')
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.title('Evolution du mAP@0.5 au cours des epochs')
plt.legend()
plt.show()


# In[ ]:


#Entraînement du Modèle YOLOv5
'''
a. Configuration du fichier d'entraînement
Crée un fichier de configuration YAML pour spécifier les chemins des données et les classes :

train: C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/ImageSets/Main/train.txt
val: C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/ImageSets/Main/val.txt
test: C:/Users/USER/Documents/tppython/VOCdevkit/VOC2012/ImageSets/Main/test.txt

nc: 20
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']










///

Affichage des résultats



