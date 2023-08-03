from datetime import datetime
from typing import Union
from pydantic import BaseModel


class ApparelBase(BaseModel):
    name: str
    width: int
    height: int
    category: str
    price: int


class ApparelCreateInput(BaseModel):
    name: str
    input64: str
    price: int = -1


class ApparelCreate(ApparelBase):
    pass


class ApparelUpdate(ApparelBase):
    pass


class Apparel(ApparelBase):
    user_id: int
    created_at: datetime
    class Config:
        orm_mode = True


class OutfitsRecommendation(BaseModel):
    """
    0: Unable recommendation
    1: Enable recommendation
    ***.jpg: Input for recommendation
    """
    top: Union[str, int]
    bottom: Union[str, int]
    bag: Union[str, int]
    outerwear: Union[str, int]
    shoe: Union[str, int]