from models import *
from utils import *
from datasets import *

import os
import sys
import time
import datetime
import argparse

import cv2
from PIL import Image

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.img_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.img_size)

    im_width, im_height = (cap.get(3), cap.get(4))

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    while True:
        print("\nPerforming object detection:")
        prev_time = time.time()

        ret, image_np = cap.read()
        # image_np = np.array(Image.open("data/samples/sample_0.jpg"))

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Configure input
        input_imgs = transforms.ToTensor()(image_np).type(Tensor)
        # print(input_imgs.shape, input_imgs)
        # input_imgs = transforms.ToTensor()(Image.open("data/samples/sample_0.jpg").convert('RGB')).type(Tensor)
        # print(input_imgs.shape, input_imgs)

        input_imgs, pad = pad_to_square(input_imgs, 0)

        input_imgs = resize(input_imgs, opt.img_size)

        input_imgs = input_imgs.unsqueeze(0)

        print(input_imgs.shape)

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            print(detections)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        detections = detections[0]

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("Inference Time: %s" % (inference_time))

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(image_np)
        print(detections)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, image_np.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig("output/video_detection.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
        cv2.imshow('Single-Threaded Detection', cv2.imread("output/video_detection.png"))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
