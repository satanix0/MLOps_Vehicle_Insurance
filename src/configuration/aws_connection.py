import boto3  # helps to connect with aws services
import os
from src.constants import ENV_AWS_SECRET_ACCESS_KEY, ENV_AWS_ACCESS_KEY_ID, REGION_NAME


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """ 
        This Class gets aws credentials from environment variables and 
        establishes a connection with s3 bucket for data and model storage
        and raises exception if environment variable are not set.
        """

        if S3Client.s3_resource == None or S3Client.s3_client == None:
            __access_key_id = os.getenv(ENV_AWS_ACCESS_KEY_ID, )
            __secret_access_key = os.getenv(ENV_AWS_SECRET_ACCESS_KEY, )
            if __access_key_id is None:
                raise Exception(
                    f"Environment variable: {ENV_AWS_ACCESS_KEY_ID} is not set.")
            if __secret_access_key is None:
                raise Exception(
                    f"Environment variable: {ENV_AWS_SECRET_ACCESS_KEY} is not set.")

            S3Client.s3_resource = boto3.resource('s3',
                                                  aws_access_key_id=__access_key_id,
                                                  aws_secret_access_key=__secret_access_key,
                                                  region_name=region_name
                                                  )
            S3Client.s3_client = boto3.client('s3',
                                              aws_access_key_id=__access_key_id,
                                              aws_secret_access_key=__secret_access_key,
                                              region_name=region_name
                                              )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
