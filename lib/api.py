from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import get_outfit_recommend

class TextInput(BaseModel):
    text: str

app = FastAPI()

config_path = "configs/outfit_recommend.yaml"
model = get_outfit_recommend(config_path)


@app.post("/outfits_recommend_from_prompt/")
async def outfit_recommend_from_prompt(prompt: TextInput):
    return model.outfits_recommend_from_prompt(prompt.text)
