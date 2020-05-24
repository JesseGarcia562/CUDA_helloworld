import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, help="filename of the output image to save")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

opt = parser.parse_known_args()[0]

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# detect objects in the image (with overlay)
detections = net.Detect(img, width, height, opt.overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))

for detection in detections:
	print(detection)

# print out timing info
net.PrintProfilerTimes()

# save the output image with the bounding box overlays
jetson.utils.saveImageRGBA(opt.file_out, img, width, height)
