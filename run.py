# Copyright 2018 Gang Wei wg0502@bu.edu

from API import use_API

use_API().load_image_path("Please enter your path to the image")
use_API().load_model_path("Please enter your path to the model")
use_API().load_label_path("Please enter your path to the label")
'''
For example:
use_API().load_image_path("/Users/gangwei/MercedesBenz.jpg")
use_API().load_model_path("/tmp/output_graph.pb")
use_API().load_label_path("/tmp/output_labels.txt")
'''
use_API().go()