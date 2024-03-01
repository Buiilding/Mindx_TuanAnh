from fastapi import FastAPI, HTTPException, Request,UploadFile
from pydantic.types import Strict
from typing import List, Union, Any
#import nest_asyncio
from pydantic import BaseModel
import traceback
from img_cls import predict
import os
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

class Item(BaseModel):
    im_path: str=None

#nest_asyncio.apply()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify(item: Item):
    try:
        return predict(item.im_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    with open("C:\\Users\\tuana\\OneDrive\\Documents\\GitHub\\storage\\" + file.filename, "wb") as f:
        f.write(file.file.read())
    absolute_path = os.path.abspath("C:\\Users\\tuana\\OneDrive\\Documents\\GitHub\\storage\\" + file.filename)
    return {"filename": absolute_path}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("api:app", host="0.0.0.0", port=8100, reload=True)
    
