"""AWS credentials service for frontend access."""

import boto3
from config.settings import get_settings


class CredentialsService:
    """Service for generating temporary AWS credentials for frontend."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._sts = boto3.client(
            "sts",
            region_name=self._settings.aws_region,
            aws_access_key_id=self._settings.aws_access_key_id or None,
            aws_secret_access_key=self._settings.aws_secret_access_key or None,
        )
        self._s3 = boto3.client(
            "s3",
            region_name=self._settings.aws_region,
            aws_access_key_id=self._settings.aws_access_key_id or None,
            aws_secret_access_key=self._settings.aws_secret_access_key or None,
        )

    def get_transcribe_credentials(self, session_name: str = "web-session") -> dict:
        """
        Get temporary credentials for AWS Transcribe Streaming.
        
        Returns credentials with limited permissions for transcribe:StartStreamTranscription.
        """
        # Policy for Transcribe Streaming only
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "transcribe:StartStreamTranscription",
                        "transcribe:StartStreamTranscriptionWebSocket",
                    ],
                    "Resource": "*",
                }
            ],
        }

        response = self._sts.get_federation_token(
            Name=session_name[:32],  # Max 32 chars
            Policy=str(policy).replace("'", '"'),
            DurationSeconds=3600,  # 1 hour
        )

        credentials = response["Credentials"]
        return {
            "accessKeyId": credentials["AccessKeyId"],
            "secretAccessKey": credentials["SecretAccessKey"],
            "sessionToken": credentials["SessionToken"],
            "expiration": credentials["Expiration"].isoformat(),
            "region": self._settings.aws_region,
        }

    def get_s3_upload_url(self, s3_key: str, content_type: str = "audio/webm") -> dict:
        """
        Generate a pre-signed URL for S3 upload.
        
        Args:
            s3_key: The S3 key for the audio file
            content_type: The content type of the file
            
        Returns:
            Dict with upload URL and the S3 key
        """
        url = self._s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._settings.s3_bucket_audio,
                "Key": s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=300,  # 5 minutes
        )

        return {
            "uploadUrl": url,
            "s3Key": s3_key,
            "bucket": self._settings.s3_bucket_audio,
        }


# Global instance
credentials_service = CredentialsService()


def get_credentials_service() -> CredentialsService:
    """Get credentials service instance."""
    return credentials_service

