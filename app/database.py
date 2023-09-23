from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


##TODO: Setup os environment for these parameters
USERNAME = "leonard"
PASSWORD = "123"
IP_ADDRESS = "localhost"
DATABASE_NAME = "mydb"

# Set up database
# SQLALCHEMY_DATABASE_URL = "postgresql://<username>:<passwoard>@<ip-address/hostname>/<database_name>"
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/{DATABASE_NAME}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
