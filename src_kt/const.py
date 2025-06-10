# about habitat scene
INVALID_SCENE_ID = []
import os
# about chatgpt api
END_POINT = "https://api.openai.com/v1/"


from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/wiss/zhang/code/openeqa/3D-Mem/.env", override=True)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")