from sqlalchemy import Column, Integer, String, Boolean, CheckConstraint
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text
from .database import Base


class Apparel(Base):
    __tablename__ = "hanger_apparels_100"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    name = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    category = Column(String, nullable=False)
    price = Column(Integer, nullable=False, default=-1)
    user_id = Column(Integer, nullable=False, default=0)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(
            category.in_(["top", "bottom", "bag", "outerwear", "shoe"]),
            name="category_check",
        ),
    )


class User(Base):
    __tablename__ = "hanger_users"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )
