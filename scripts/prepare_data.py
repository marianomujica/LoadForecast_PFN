from app.config import load_settings
from app.service import prepare_all

if __name__ == "__main__":
    print(prepare_all(load_settings()))
