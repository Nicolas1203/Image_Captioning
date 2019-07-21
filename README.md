Image Captioning
================

Image captioning pet project, based on Udacity's Computer Vision Nanodegree project.

How to run
----------

After downloading coco dataset and installing the requirements, you can run `python3 run.py -h`
to display the available options.

Training a model
----------------
TODO

Visualizing results on coco test
--------------------------------
TODO


Requirements
------------

Run `pip3 install -r requirements.txt` to install the required python packages

This repo has been tested on Ubuntu 16.04, 18.04 and Elementary OS 5.0 Juno.

If you want to use GPU, please be sure that you have CUDA installed.

COCO dataset
------------

* First, clone the [coco api repository](https://github.com/cocodataset/cocoapi) into `/opt` directory
* Then, go to [coco website](http://cocodataset.org/#download) and download the 2014 test and train images.
* Create an image folder and an annotation folder: `mkdir /opt/cocoapi/images` and `mkdir /opt/cocoapi/annotations`.
* Unzip the train and test images into `/opt/cocoapi/images`.
* Unzip the train and test annotations into `/opt/cocoapi/annotations`

