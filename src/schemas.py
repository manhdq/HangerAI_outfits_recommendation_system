from pydantic import BaseModel
from typing import List


class ApparelsInput(BaseModel):
    text: str
    name: List[str]
    input64: List[str]
