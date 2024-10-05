import genai_model
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
#import json

app = FastAPI()

class Item(BaseModel):
    values: List[float]

@app.post("/genai")
async def genai(size_output: int, item: Item):
    # Creamos un modelo generativo 
    input_data = item.values
    size_input = len(input_data)
    model = genai_model.GenerativeModel(size_input, size_output)

    # Generamos datos a partir del modelo
    output_data = model.generate(input_data)

    str = f"Datos generados: {output_data}"

    return str