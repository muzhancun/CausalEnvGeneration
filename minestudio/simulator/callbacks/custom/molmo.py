import base64
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
from io import BytesIO

def encode_image_base64(image: np.array) -> str:
    # Convert the image to a base64 string
    img = Image.fromarray(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return img_base64


class Pointer:
    '''
    Pointer is a model based on molmo, which can output point location of the object in the image.
    '''
    def __init__(self, model_id, model_url:str="http://127.0.0.1:9162/v1", api_key:str="EMPTY"):
        self.model_id = model_id
        self.model_url = model_url
        self.api_key = api_key

    def post_init(self):
        if self.model_url == None or self.model_url == 'huggingface':
            self.client = None
            self.load_molmo_from_hf(self.model_id)
        else:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.model_url,
            )
            models = client.models.list()
            print(models)
            model = models.data[0].id
            assert model == self.model_id, f"Model {self.model_id} not found in current model_url {self.model_url}"
            print(f"Using model {self.model_id} based on url {self.model_url}")
            self.client = client

    def load_molmo_from_hf(self, model_id):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto')
        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto')

    def openai_generation(self, prompt:str, image) -> str:
        if hasattr(self, "client") == False:
            print("Initializing OpenAI client")
            self.post_init()
        image_base64 = encode_image_base64(image)
        chat_completion_from_base64 = self.client.chat.completions.create(
            messages=[{
                "role":"user",
                "content": [{"type": "text", "text": prompt},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            model=self.model_id,
        )
        return chat_completion_from_base64.choices[0].message.content

    def parse_coordinates(self, xml_string: str):
        # Parse the XML string
        root = ET.fromstring(xml_string)
        # Initialize an empty list to store coordinates
        coordinates = []
        # Iterate over the attributes of the XML node
        for attr_name, attr_value in root.attrib.items():
            # Check if the attribute is an 'x' or 'y' coordinate by matching the pattern
            if attr_name.startswith('x'):
                # Get the corresponding 'y' coordinate
                y_attr_name = 'y' + attr_name[1:]  # Assume 'y' coordinate has the same index
                if y_attr_name in root.attrib:
                    # Append the (x, y) tuple to the coordinates list
                    coordinates.append((float(attr_value), float(root.attrib[y_attr_name])))
        return coordinates

    # def gen_point(self, image:Image, object_name:str):
    def gen_point(self, image:Image, prompt:str):
        # prompt = f"Check whether {object_name} in this image. If {object_name} exists in the image, pinpoint all {object_name}. If not, output 'None'."
        self.molmo_result = self.openai_generation(prompt, image)
        if self.molmo_result == "NONE":
            return None
        print("Pointing result: ", self.molmo_result)
        if 'none' in self.molmo_result.lower():
            return None
        else:
            points = self.parse_coordinates(self.molmo_result)
            points = [(int(x/100*640), int(y/100*360)) for x, y in points]
            return points 
