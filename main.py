import os
import uuid
from fastapi import BackgroundTasks, Body, FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
from segSam import segMobileSam, segSam2_b_plus, segSam2_b_plus_cached, segSam2_b_plus_remove, segSam2_small, segSam2_tiny
import cv2
import logging
from redis.asyncio import Redis
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import secrets


redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins if necessary
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


async def verify_api_key(api_key: str = Form(...)):
    """
    Verifies if the provided API key exists in Redis.
    Raises an HTTPException if the key is invalid.
    """
    stored_api_key = await redis_client.get("api_key")
    if api_key != stored_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    # api_keys = await redis_client.lrange("api_key", 0, -1)
    # if api_key not in api_keys:
        # raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")


@app.post("/generate_api_key/")
async def generate_api_key_view():
    api_key = secrets.token_hex(32)
    await redis_client.set("api_key", api_key)
    # await redis_client.lpush("api_key", api_key)
    return JSONResponse({"api_key": api_key})


@app.post("/upload/")
async def upload_image(
    background_tasks: BackgroundTasks, 
    image: UploadFile = File(...), 
    coords: str = Body(...), 
    option: str = Body(...), 
    api_key: str = Depends(verify_api_key)):
    """
    API endpoint to accept an image file and return it in the response.
    """
    
    logger.debug(f"data: {coords}")
    data = convert_to_2d_list(coords)
    logger.debug(f"data: {data}")
    file_extension = image.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"./uploads/{file_name}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
 
    if (option == "1" or option == "4"):
        result = segSam2_small(file_path, data, bool(int(option) % 2))
    elif option == "2" or option == "5":
        result = segSam2_tiny(file_path, data, not bool(int(option) % 2)) 
    elif option == "3" or option == "6":
        result = segSam2_b_plus(file_path, data, bool(int(option) % 2))  
    elif option == "7" or option == "8":
        result = segSam2_b_plus_remove(file_path, data, bool(int(option) % 2))           
    else:
        result = segMobileSam(file_path, data)

    file_name = os.path.basename(result)
 
    background_tasks.add_task(cleanup, result)
    return FileResponse(path=result, filename=file_name)

@app.post("/cached/")
async def upload_image(
    background_tasks: BackgroundTasks, 
    image: UploadFile = File(...), 
    coords: str = Body(...), 
    option: str = Body(...), 
    uuid_key: str = Body(...),
    api_key: str = Depends(verify_api_key)):
    """
    API endpoint to accept an image file and return it in the response.
    """
    uuid_part = os.path.basename(uuid_key).split('.')[0]
    logger.debug(f"uuid_partp--: {uuid_part}")
    data = convert_to_2d_list(coords)
    file_extension = image.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"./uploads/{file_name}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    result = segSam2_b_plus_cached(file_path, data, bool(int(option) % 2),uuid=uuid_part)  

    file_name = os.path.basename(result)

    background_tasks.add_task(cleanup, result)
    return FileResponse(path=result, filename=f"{file_name}.png")

@app.get("/clean")
async def root():
    clear_directory('./result')
    clear_directory('./uploads')

@app.get("/")
async def root():
 
    return {"message": "PONG"}

    # content = await image.read()
    # filename = f"{str(uuid.uuid4())}.png"
    # # Save the uploaded file to the local directory
    # with open(file_location, "wb") as buffer:
    #     shutil.copyfileobj(image.file, buffer)
    
    # Return the image as a response (FileResponse returns a file as a response)


def convert_to_2d_list(input_str):
    # Ensure the input is a string and has a valid format
    if not isinstance(input_str, str) or not (input_str.startswith("[") and input_str.endswith("]")):
        raise ValueError("Input should be a valid list format as a string.")

    # Step 1: Remove the outer brackets
    cleaned_str = input_str[1:-1].strip()  # Remove the outer [ and ]

    # Step 2: Split the string into individual inner lists
    inner_lists = cleaned_str.split('],[')

    # Step 3: Convert each inner list string to a list of integers
    data = []
    for inner_list in inner_lists:
        # Remove any extra brackets and split by comma
        numbers = inner_list.replace('[', '').replace(']', '').split(',')
        # Convert the numbers to integers, strip whitespace, and filter out empty strings
        data.append([int(num.strip()) for num in numbers if num.strip()])

    return data

def clear_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate through all the items in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                # Check if it's a file or directory and remove accordingly
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory and all its contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        print(f"The directory {directory} does not exist.")

        
def cleanup(temp_file):
    os.remove(temp_file)