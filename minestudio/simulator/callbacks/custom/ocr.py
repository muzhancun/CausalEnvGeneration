import cv2

def process_image(img):
    # import ipdb; ipdb.set_trace()
    img = img[180:360, 0:330]
    # thresh_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # # 提高对比度
    # thresh_img = cv2.convertScaleAbs(thresh_img, alpha=1.5, beta=0)

    # # 二值化
    # _, thresh_img = cv2.threshold(thresh_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 放大图片
    # thresh_img = cv2.resize(thresh_img, (thresh_img.shape[1] * 3, thresh_img.shape[0] * 3))

    return img

def get_string_gvision(img):
    # use google cloud vision to extract text from image
    from google.cloud import vision
    import io
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'minestudio/simulator/callbacks/custom/ocr.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img.tobytes())
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description

def get_string(img, config = r'--psm 10'):
    import pytesseract
    string = pytesseract.image_to_string(img, lang='eng', config = config)
    return string

def get_string_paddle(img_path):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='en') # need to run only once to load model into memory
    result = ocr.ocr(img_path, cls=False)
    if result is None:
        return ''
    str = ''
    print(result)
    for idx in range(len(result)):
        res = result[idx]
        if res is None:
            continue
        for line in res:
            str += line[1][0]
    return str

def clean_string(string):
    tmp = ''
    for line in string:
        tmp += line.replace('\n', ' ')
    tmp = tmp.replace("'",'"')
    return tmp

if __name__ == '__main__':
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        PlayCallback
    )
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            PlayCallback()
        ]
    )
    obs, info = sim.reset()
    terminated = False
    obs, reward, done, info = sim.env.execute_cmd("/gamemode creative")
    obs, reward, done, info = sim.env.execute_cmd("/effect give @s minecraft:night_vision 99999 1 true")
    obs, reward, done, info = sim.env.execute_cmd("/locate village")
    # import ipdb; ipdb.set_trace()
    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(action)
        # import ipdb; ipdb.set_trace()
        image = info['pov']
        img = process_image(image)
        # save image
        cv2.imwrite('./test.png', img)
        string = get_string_paddle('./test.png')
        # string = get_string(img)
        print(string)