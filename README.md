# gjenbruksstasjoner-kotid-merking
Needed to label images to be used for model training. It has been tested in Ubuntu 20.04.

## Installation
It is recommended to work in a virtual environment, e.g. by using the `venv` module, before installing the requirements.  
`$ python3 -m venv .venv`  
`$ source .venv/bin/activate` (when done, `deactivate` the venv).  

Install the required libraries:  
`$ pip install -r requirements.txt`  


## Download images to local destination folder
You need to be logged in to the AWS-account (contact Origo for the account number) with access to the right S3 buckets to download the images. Use `saml2aws` or similar tools.  
Modify the HARDCODED PARAMETERS in `download_images.py` to reflect the bucket and location of the file, as well as the target directory when downloading.  
Run `python3 download_images.py`.

## Label the images
Run `python3 label_images.py`.  
See the instructions which ar immediately printed. Or press `h` for help.
