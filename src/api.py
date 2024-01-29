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


## TODO
# async def outfit_recommend_from_prompt(data_input: ApparelsInput):
#     prompt = data_input.text
#     item_names = data_input.name
#     input64s = data_input.input64

#     storage = {}
#     for name, inp64 in zip(item_names, input64s):
#         image = base64_to_image(inp64)

#     return model.outfits_recommend_from_prompt(prompt.text)
