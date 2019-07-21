Image Captioning
================

Image captioning pet project, based on Udacity's Computer Vision Nanodegree project.

How to run
----------

After downloading coco dataset and installing the requirements, you can run `python3 run.py -h`
to display the available options.

Training a model
----------------
`python3 run.py --train --epochs 10 --batch-size 16 --log-file training_logs.txt`

Visualizing results on coco test
--------------------------------
If you have an encoder and decoder saved using torch save function:
`python3 run.py --inference-coco --encoder-file encoder.pkl --decoder-file decoder.pkl`

Requirements
------------

Run `pip3 install -r requirements.txt` to install the required python packages

This repo has been tested on Ubuntu 16.04, 18.04 and Elementary OS 5.0 Juno.

If you want to use GPU, please be sure that you have CUDA installed.

Example
-------
![](example.png)

Prediction: `a display case filled with lots of donuts .`


COCO dataset
------------

* First, clone the [coco api repository](https://github.com/cocodataset/cocoapi) into `/opt` directory
* Then, go to [coco website](http://cocodataset.org/#download) and download the 2014 test and train images.
* Create an image folder: `mkdir /opt/cocoapi/images`
* Unzip the train and test images into `/opt/cocoapi/images`.
* Unzip the train and test annotations into `/opt/cocoapi/`

Your `/opt/cocoapi` directory should look like:
```bash
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── image_info_test2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
├── common
│   ├── gason.cpp
│   ├── gason.h
│   ├── maskApi.c
│   └── maskApi.h
├── images
│   ├── test2014
│   └── train2014
├── license.txt
├── LuaAPI
│   ├── CocoApi.lua
│   ├── cocoDemo.lua
│   ├── env.lua
│   ├── init.lua
│   ├── MaskApi.lua
│   └── rocks
├── MatlabAPI
│   ├── CocoApi.m
│   ├── cocoDemo.m
│   ├── CocoEval.m
│   ├── CocoUtils.m
│   ├── evalDemo.m
│   ├── gason.m
│   ├── MaskApi.m
│   └── private
├── PythonAPI
│   ├── Makefile
│   ├── pycocoDemo.ipynb
│   ├── pycocoEvalDemo.ipynb
│   ├── pycocotools
│   └── setup.py
├── README.txt
└── results
    ├── captions_val2014_fakecap_results.json
    ├── instances_val2014_fakebbox100_results.json
    ├── instances_val2014_fakesegm100_results.json
    ├── person_keypoints_val2014_fakekeypoints100_results.json
    └── val2014_fake_eval_res.txt
```

TODO:
----
* Add resume training option
* Add video processing support


