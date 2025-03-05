import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from app.config import settings
from app.utils.logger import logger

class S3Storage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3_bucket_name

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for secure access to an S3 object.
        Expires after `expiration` seconds (default: 1 hour).
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            return url
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"S3 presigned URL generation failed: {str(e)}")
            raise

    async def upload_media(self, file_path: str, object_key: str) -> bool:
        """
        Upload a media file to S3.
        """
        try:
            with open(file_path, 'rb') as file:
                self.s3_client.upload_fileobj(file, self.bucket_name, object_key)
            logger.info(f"Successfully uploaded {object_key} to S3")
            return True
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return False