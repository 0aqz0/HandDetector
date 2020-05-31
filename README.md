# HandDetector
Hand detector with yolov3

<img src='output/hand_detection.gif' width="300">

## Requirements

- Install packages

  ```bash
  $ pip install -r requirements
  ```

- Data preparation

  ```bash
  $ python get_egohands_data.py
  ```

## Usage

**Training**

```bash
$ python train.py --pretrained_weights weights/darknet53.conv.74
```

**Testing**

```bash
$ python test.py --weights_path weights/yolov3.pth
```

**Detection with images**

```bash
$ python detect.py --image_folder data/samples/ --weights_path weights/yolov3.pth
```

**Detection with camera**

```bash
$ python video_detect.py --weights_path weights/yolov3.pth
```

## Reference

- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- [victordibia/handtracking](https://github.com/victordibia/handtracking)