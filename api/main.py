from fastapi import FastAPI
import base64
from PIL import Image
import onnxruntime as ort
from fastapi import UploadFile, File
import io
import onnxruntime as rt
from preprocess_image import generar_datos 
import numpy as np

app = FastAPI()

@app.get("/")
async def health_check():
    return "The api is working?"

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    input_tensor = generar_datos(image)
    model = ort.InferenceSession('random_forest_model.onnx')
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: input_tensor.numpy()})
    return {"prediction": output[0][0]}

@app.post("/predict_from_data")
async def predict_from_data(data: dict):
    input_tensor = np.array([list(map(float, data.values()))], dtype=np.float32)
    model = ort.InferenceSession('random_forest_model.onnx')
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: input_tensor})
    return {"prediction": output[0][0]}

@app.get("/test")
async def test():
    session = rt.InferenceSession("random_forest_model.onnx")
    nuevo_dato = np.array([[1, 3.2, 0.42, 0.89, 0.80, 0.99, 30, 34, 22, 
                        0.81, 0.0003, 0.0001, 0.0002, 0.00007, 0.00003, 
                        0.00039, 0.00013, 0.00066, 0.00013, 0.000297, 0.000250, 0.000297, 0.000297]], dtype=np.float32)
    # Realizar la predicción
    prediccion = session.run([output_name], {input_name: nuevo_dato})
    return f"Predicción: {prediccion[0][0]}"