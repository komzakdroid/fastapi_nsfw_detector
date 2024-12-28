from fastapi import FastAPI, File, UploadFile
from nsfw_detector import predict

app = FastAPI()
model = predict.load_model("nsfw_model.h5")

@app.post("/moderate/")
async def moderate_image(file: UploadFile = File(...)):
    image = await file.read()
    result = predict.classify(model, image)
    return {"safe": result.get("neutral", 0) > 0.8}
