# api/auth.py
import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

API_SECRET_KEY = os.environ.get("API_SECRET_KEY")


async def verify_api_key(x_api_key: str = Header(...)):
    """
    Dependency that validates the X-Api-Key header on every protected route.
    Inject with: dependencies=[Depends(verify_api_key)]
    """
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API_SECRET_KEY is not set."
        )
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Pass your key in the X-Api-Key header."
        )
