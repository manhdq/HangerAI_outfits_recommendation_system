from fastapi import FastAPI
from pydantic import BaseModel
from .schemas import ApparelsInput
from .pipeline import get_outfit_recommend

class TextInput(BaseModel):
    text: str

app = FastAPI()

config_path = "configs/polyvore_outfit_recommend.yaml"
fhn_config_path = "configs/FHN_VOE_T3_fashion32.yaml"
model = get_outfit_recommend(config_path)


@app.post("/outfits_recommend_from_prompt/", status_code=200)
async def outfit_recommend_from_prompt(prompt: TextInput):
    return model.outfits_recommend_from_prompt(prompt.text)


## TODO
@app.post("/outfits_recommend_from_inputs/", status_code=200)
async def outfit_recommend_from_inputs(
    # data_input: ApparelsInput
    prompt: TextInput
):
    # prompt = data_input.text
    # item_names = data_input.name
    # input64s = data_input.input64

    return model.outfits_recommend_from_inputs(
        fhn_config_path,
        prompt,
        # item_names,
        # input64s
    )
