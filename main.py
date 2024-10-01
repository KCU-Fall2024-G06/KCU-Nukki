import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
from seg import segImage
from segSam import segSam2
import cv2
import logging

app = FastAPI()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)
@app.post("/upload/")
async def upload_image(image: UploadFile = File(...)):
    """
    API endpoint to accept an image file and return it in the response.
    """
    file_extension = image.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"./uploads/{file_name}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    logger.debug(f"path: {file_path}")
    result = segSam2(file_path)
    # content = await image.read()
    # filename = f"{str(uuid.uuid4())}.png"
    # # Save the uploaded file to the local directory
    # with open(file_location, "wb") as buffer:
    #     shutil.copyfileobj(image.file, buffer)
    
    # Return the image as a response (FileResponse returns a file as a response)
    return FileResponse(path=result, filename="result.png")

@app.get("/")
async def root():
    img =  cv2.imread("./uploads/test.png",cv2.IMREAD_UNCHANGED)
    result = segImage(img)
    return {"message": result}