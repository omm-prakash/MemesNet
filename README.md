# MemesNet
A multi-modality deep learning model to detect hateful memes. The code is submitted as a solution to the task from Prof. Ponnurangam Kumaraguru's [Precog Lab](https://precog.iiit.ac.in/) @IIIT Hyderabad. 

## Description
The [Hateful Memes Challenge](https://hatefulmemeschallenge.com/#download) and Dataset have been developed to advance research in identifying harmful multimodal content, particularly focusing on the challenging task of recognizing multimodal hate speech, which combines different modalities such as text and images. This unique dataset, created by Facebook AI, consists of over 10,000 new multimodal examples, incorporating content licensed from Getty Images to provide a diverse and comprehensive resource for AI researchers.

## Environment Setup

### Conda Setup
To ensure a consistent environment, it is recommended to use conda for managing dependencies. Follow these steps to set up the environment using the provided `environment.yml` file.

1. Clone the repository:
   ```bash
   git clone https://github.com/omm-prakash/MemesNet.git
   cd MemesNet
   ```

2. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate memes
   ```

### Dataset Download
Download the dataset from [Hateful Memes Challenge](https://hatefulmemeschallenge.com/#download) and follow the steps below to organize your project structure:

1. Create a `data` directory in the project root:
   ```bash
   mkdir data
   ```

2. Save the downloaded dataset file into the `data` directory.

3. Unzip the dataset:
   ```bash
   unzip data/your-dataset.zip -d data/
   ```

Now the environment is set up, and the dataset is ready for use in the project.

### Usage

#### 1. Object Detection

To perform object detection on images using YOLO (You Only Look Once) model, run the following command:

```bash
python object_detection.py \
  --directory ./data/hateful_memes/img/ \
  --file <path_to_image_file> \
  --pick_random <True/False> \
  --plot <True/False> \
  --config data/yolov3.cfg \
  --weight data/yolov3.weights \
  --classes data/coco.names
```

- `--directory`: Path of the folder containing all images.
- `--file`: Image file path (optional, if not using random image).
- `--pick_random`: Select a random image for detection (default: True).
- `--plot`: Show the image with detected objects (default: True).
- `--config`: Path for YOLO model configuration file (default: data/yolov3.cfg).
- `--weight`: Path for YOLO model weight file (default: data/yolov3.weights).
- `--classes`: Path for COCO class name file (default: data/coco.names).

#### 2. Optical Character Recognition (OCR)

To extract text from images using OCR, use the following command:

```bash
python ocr.py \
  --directory ./data/hateful_memes/img/ \
  --file <path_to_image_file> \
  --draw_anno <True/False> \
  --pick_random <True/False> \
  --plot <True/False>
```

- `--directory`: Path of the folder containing all images.
- `--file`: Image file path (optional, if not using random image).
- `--draw_anno`: Draw annotations on the image (default: False).
- `--pick_random`: Select a random image for OCR (default: True).
- `--plot`: Show the image with extracted text (default: True).

#### 3. Training

To initiate the training process with a modified configuration in `config.yml,` execute the following command:

```bash
python train.py
```
Ensure that you have updated the necessary configurations in the `config.yml` file before starting the training process.

## License
The project has not been licensed till now.

## Acknowledgements
The cross-modality encoder part is referred from [LXMERT](https://arxiv.org/abs/1908.07490). 

## Contact
Please contact me at ommprakash.sahoo.eee20@iitbhu.ac.in for any query related to the code.
