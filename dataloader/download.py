import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config




def download_raw_data(files, zip_data_path, bucket_name, file_key_template):
    # Create an unsigned S3 client for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for file in files:
        file_key = os.path.join(file_key_template, file)
        download_path = os.path.join(zip_data_path, file)
        
        # Download the file
        print(f"Downloading file to {download_path}")
        s3.download_file(bucket_name, file_key, download_path)
        print(f"Finished")


zip_data_path = 'data/zip_data'

# Create the data directory if it doesn't exist
os.makedirs(zip_data_path, exist_ok=True)


# Specify bucket and file details
file_key_template = 'spacenet/SN8_floods/tarballs'
bucket_name = 'spacenet-dataset'
files = [
    'Germany_Training_Public.tar.gz',
    'Louisiana-East_Training_Public.tar.gz',
    'Louisiana-West_Test_Public.tar.gz',
]

download_raw_data(files, zip_data_path, bucket_name, file_key_template)