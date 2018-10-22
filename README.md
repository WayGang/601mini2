# 601mini2
MLrecognize
 
I trained this system by thousands of images of SUVs and sedans.

But the model is too big to upload here. 

You can train it by your own dataset by the following steps.

Your dataset should have some subfolders, which contains different kinds of images.

For example, all suv images should be in a subfolder named "suv", 

and all sedan images should be in a subfolder named "sedan".


Also, I provided a API to implement handily.


retrain:

10% of the dataset will be used for testing, and another 10% will be used for validation. 

To use your own dataset to train the model, just add the path of your dataset to the end of the terminal command.

label_image:(if you don't wanna use API)

You will enter the path of your image, graph and label to recognize the image.

API:

You can download the "API.py".

And check "run.py" ,it shows the example of how to use the API.
