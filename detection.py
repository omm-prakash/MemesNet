import cv2, os
from pytorchyolo import detect, models
import random
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def load_mode(config, weight):
        model = models.load_model(config, weight)
        return model

def load_class(path):
        with open(path, "r") as fp:
                names = fp.read().splitlines()
        return names

def extract_data(box):
        x1, y1, x2, y2, confidence, class_id = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        return {'area':area, 'confidence':confidence, 'class_id':class_id}

def plot_object(image, boxes, names):
        for i,box in enumerate(boxes):
                x1, y1, x2, y2, confidence, class_id = box
                color = (255,0, 255)
                data = extract_data(box)
                print(f'   {i+1}. class: {names[int(class_id)]}, confidence: {confidence*100:.4f}%, area: {data["area"]:.4f} sq.px')
                label = f"{names[int(class_id)]} @{confidence:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.axis(False)
        plt.show()
        if not len(boxes):
                print('>> Sorry, no object found.')
        return 

def parser():
        parser = argparse.ArgumentParser(description='This code helps in object detection in the image.')
        parser.add_argument('--directory', type=str, default="./data/hateful_memes/img/", help='Path of folder having all images.')
        parser.add_argument('--file', type=str, default=None, help='Image file path.')
        parser.add_argument('--pick_random', type=bool, default=True, help='Select random image.')
        parser.add_argument('--plot', type=bool, default=True, help='Show image.')
        parser.add_argument('--config', type=str, default='data/yolov3.cfg', help='Path for YOLO model config file.')
        parser.add_argument('--weight', type=str, default='data/yolov3.weights', help='Path for YOLO model weight file.')
        parser.add_argument('--classes', type=str, default='data/coco.names', help='Path for COCO class name file.')

        args = parser.parse_args()
        return args

def detect_object(args):
        print('\n>> Loading test image..')
        if args.file is not None:
                f = os.path.join(args.directory, args.file)
        elif args.pick_random:
                image_files = os.listdir(args.directory)
                f = os.path.join(args.directory, random.choice(image_files))
        else:
                print('Please pass name of image file or allow random pick of image.')

        print('>> Image path:', f)
        print('>> Loading model..')
        model = load_mode(args.config, args.weight)

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('>> Detecting objects..')
        boxes = detect.detect_image(model, img)
        classes = load_class(args.classes)
        print('>> Drawing objects..')
        plot_object(img, boxes, classes)
        return

if __name__=='__main__':
        args = parser()
        detect_object(args)
        