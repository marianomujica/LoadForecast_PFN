from app.config import load_settings
from app.service import train_one

if __name__ == "__main__":
    print(train_one(load_settings(), 1))
