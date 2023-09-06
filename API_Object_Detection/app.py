import base64
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import *
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import shutil

def delete_dir_contents(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def send_file_data(filepath):
    img = Image.open(filepath)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = buffered.getvalue()
    return img_str

def send_file_data_2(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes)
    return Response(content=encoded_image, headers={"Content-Type": "image/jpg"})

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    dir_path = './runs/detect'
    delete_dir_contents(dir_path)
    image = read_file_as_image(await file.read())
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    model = YOLO('best.pt')
    pred = model.predict(image ,save=True, save_crop=True , save_conf=True , save_txt=True)
    filename = 'runs/detect/predict/image0.jpg'
    label_all = []
    conf = []
    if not ("crops" in os.listdir("./runs/detect/predict")) :
          label_all.append({'image' : send_file_data_2(filename),
                            'class' : "Not Found",
                            'conf'  : "100 %"})
    else :
        for file_result in os.listdir("./runs/detect/predict/labels"):
            file_result_path = "./runs/detect/predict/labels"+ "/" + file_result
            with open(file_result_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    last_element = line.split()[-1]
                    conf.append(str(int(round(float(last_element),2)*100))+" %")
        idx = 0
        for catagory in os.listdir("./runs/detect/predict/crops"):
                for label_img in os.listdir(f"./runs/detect/predict/crops/{catagory}"):
                      file_name_label = f"./runs/detect/predict/crops/{catagory}/{label_img}"
                      label_all.append({
                                        'image' : send_file_data_2(file_name_label),
                                        'class' : catagory ,
                                        'conf'  : conf[idx] 
                                      })
                      idx += 1
        label_all.append({'image' : send_file_data_2(filename),
                          'class' : "Found",
                          'conf' : "100 %"})
    return label_all
                      
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
