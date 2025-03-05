from fastapi import APIRouter, HTTPException
from app.utils.s3_storage import S3Storage
from app.utils.logger import logger

router = APIRouter(tags=["Media"])
s3_storage = S3Storage()

@router.get("/media/{object_key}")
async def get_media(object_key: str):
    """
    Generate a presigned URL for secure access to an S3 object.
    Expires after 1 hour (3600 seconds).
    """
    try:
        signed_url = s3_storage.generate_presigned_url(object_key)
        return {"url": signed_url}
    except Exception as e:
        logger.error(f"Media retrieval failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Media not found")