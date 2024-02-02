import os,random
import matplotlib.pyplot as plt
import keras_ocr
import numpy as np
import warnings
from functools import cmp_to_key, partial
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
pipeline = keras_ocr.pipeline.Pipeline()

def comparator(i,j, location):
    x1, y1 = location[i]
    x2, y2 = location[j]
    if abs(y2 - y1) < 6:
        return x1 - x2
    return y1 - y2

def detect_text(path):
    try:
        image = [keras_ocr.tools.read(path)]
    except:
        raise "Error is loading image file."
    group = pipeline.recognize(image)

    index = list(range(len(group[0])))
    location = np.array(list(map(lambda x: x[1].mean(axis=0), group[0])))
    fixed_compare = partial(comparator, location=location)
    sorted_index = sorted(index, key=cmp_to_key(fixed_compare))
    text = ' '.join(map(lambda x: group[0][x][0], sorted_index))

    return text, group[0], image[0]

def parser():
    parser = argparse.ArgumentParser(description='The code help reading text from image.')
    parser.add_argument('--directory', type=str, default="./data/hateful_memes/img/", help='Path of folder having all images.')
    parser.add_argument('--file', type=str, default=None, help='Image file path.')
    parser.add_argument('--draw_anno', type=bool, default=False, help='Check for if to draw annotations.')
    parser.add_argument('--pick_random', type=bool, default=True, help='Select random image.')
    parser.add_argument('--plot', type=bool, default=True, help='Show image.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    if args.file is not None:
        f = os.path.join(args.directory, args.file)
    elif args.pick_random:
        image_files = os.listdir(args.directory)
        f = os.path.join(args.directory, random.choice(image_files))
    else:
        print('Please pass name of image file or allow random pick of image.')
    print('\n>> Image location:',f)
    print('>> Detecting text from the image..')
    text, locations, image = detect_text(f)

    if args.plot:
        plt.imshow(plt.imread(f))
        plt.axis(False)
        print(f'>> Detected Text: "{text}"')

    if args.draw_anno:
        keras_ocr.tools.drawAnnotations(image=image, predictions=locations)
    plt.show()
