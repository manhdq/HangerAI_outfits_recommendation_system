from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import get_outfit_recommend

class TextInput(BaseModel):
    text: str

app = FastAPI()

config_path = "configs/polyvore_outfit_recommend.yaml"
model = get_outfit_recommend(config_path)


@app.post("/outfits_recommend_from_prompt/", status_code=200)
async def outfit_recommend_from_prompt(prompt: TextInput):
    return model.outfits_recommend_from_prompt(prompt.text)
