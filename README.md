# gjenbruksstasjoner-kotid-merking
Needed to label images to be used for model training.

# Download images to local destination folder
Install the required libraries:  
`$ pip install -r requirements.txt`
You need to be logged in to an AWS-account with access to the right S3 buckets to make this work. Use `saml2aws` or similar tools.   
Modify the HARDCODED PARAMETERS in `download_images.py`.  
Run `python3 download_images.py`.

# Label the images
Run `python3 label_images.py`.
See the instructions which are immediately printed. Or press `h` for help.

