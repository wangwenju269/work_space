from core_agent.tools import Tool
import time
from random import choice
from hashlib import md5
from string import digits, ascii_letters
import requests
import urllib.request
import cv2
import base64
import numpy as np
import requests
import json  
import re 
import os 

class safety_belt_checker(Tool):
    description =  "The service analyzes images to assess whether a worker is wearing a safety belt, identifies the worker's location, and verifies the use of safety equipment."
    name = 'safety_belt_checker'
    parameters: list =  [
                {
                    "name": "image",
                    "type": "string",
                    "description": "'image' must exist in the specified file path, and it should have '.png' or '.jpg' suffix indicating its file format. eg: xxx.jpg or xxx.png",
                    'required': True
                }
            ]
        
    def __init__(self, cfg={}) -> None:
        super().__init__(cfg)
        self.access_token_url =   self.cfg.get('access_token_url', '')
        self.api_key =  self.cfg.get('api_key', '') 
        self.api_secret = self.cfg.get('api_secret', '')   
        self.token_url =   self.cfg.get('token_url', '')   
        self.save_folder =  self.cfg.get('save_folder', '')  
        self.read_folder =  self.cfg.get('read_folder', '') 

    def get_token(self):
        """
        generate token
        """
        def generate_md5(src):
            m = md5(src.encode(encoding='utf-8'))
            return m.hexdigest()

        def generate_random_str(randomlength=24):
            str_list = [choice(digits + ascii_letters) for i in range(randomlength)]
            random_str = ''.join(str_list)
            return random_str

        nowtime = time.time()
        timestamp = str(int(nowtime * 1000))
        noncestr = generate_random_str(24)
        raw_token = self.api_key + ":" + timestamp + ":" + noncestr + ":" + self.api_secret
        auth = generate_md5(raw_token)
        payload = {}
        headers = {
            'X-AIOT-APIKEY': self.api_key,
            'X-AIOT-TIMESTAMP': timestamp,
            'X-AIOT-NONCESTR': noncestr,
            'Authorization': "Basic " + auth
        }
        try:
            response = requests.request("POST", self.access_token_url, headers=headers, data=payload)
            if response.status_code == 200:
                res = response.json()
                if res["code"] != 200:
                    raise Exception(res["message"])

                token = "Bearer " + res["data"]["accessToken"]
                return (token)
            else:
                raise Exception(response.status_code)
        except Exception as err:
            print('An exception happened: ' + str(err))
            return ""
        
    def download_cv_image(self, img_url):
        req = urllib.request.Request(
                            img_url,
                            data=None,
                            headers={
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                                     },
                            # headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
                            )
        with urllib.request.urlopen(req) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image_as_cvimage = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image_as_cvimage

    def cv_image2string(self, image_to_encode):
        # Encode opencv compatible image to base64 format for transmitting
        retval, buffer = cv2.imencode('.jpg', image_to_encode)
        image_as_text = base64.b64encode(buffer)
        return image_as_text

    def base64_to_cv2(self, b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data


    def construct_input(self, img_data):
        url_data = img_data.pop('url', '')
        img_base64 = img_data.pop('base64', '')
        img_cv = img_data.pop('image', '')
        img_path = img_data.pop('image_path', '')
        if  img_path:
            img_cv = cv2.imread(img_path)
            img_base64 = self.cv_image2string(img_cv) 
            img_base64 = str(img_base64)[2:-1] 

        elif  url_data:
            img_cv = self.download_cv_image(url_data)
            img_base64 = self.cv_image2string(img_cv)
            img_base64 = str(img_base64)[2:-1] 

        elif img_cv:
            img_base64 = self.cv_image2string(img_cv)

        data = {
               "img_base64":[img_base64]
               }
        return data
    
    def _remote_call(self, *args, **kwargs):         #(self, image_path, show_file):
        image_files = kwargs.get('image')
        re_image_files = re.search(r'<(.*)>', image_files)
        if  re_image_files:  image_files = re_image_files.group(1)
        image_path = os.path.join(self.read_folder,image_files)  
        "先支持图片的案例,后续有需求，增加 Url"
        data =  {'image_path':image_path} 
        token = self.get_token()
        headers = {
                "Authorization":token,
                "User-Agent":"PostmanRuntime/7.26.10",
                "Content-Type":"application/json"
            }
        data = self.construct_input(data)
        response = requests.post(self.url,headers=headers,data=json.dumps(data),stream=True).json()
        # 保存 API 返回结果的所保留文件信息
        save_file = os.path.join(self.save_folder,f'output_{image_files}')  
        self.save_img(response, data['img_base64'][0], save_file)
        response = self.post_process(response)
        return {'result': response}
    
    def post_process(self,response):
        nature_language  = ""
        if response['code']!=200:
           nature_language += 'Failed to analies the image, please check if your image or url is available'
        else:
            results = response["data"][0]
            if len(results) == 0:    
                nature_language += 'There are no workers performing high-altitude operations without wearing safety belt in the image.'
            else:
                stat = ''
                for idx, res in enumerate(results):
                    point = res['boundingCoordinates'][0][0]
                    posib = round(res['confidence'],2)*100
                    stat += f'\nworker{idx}, located at coordinates {point}, has a {posib}% probability of not wearing a safety belt.'
                nature_language += f'There are {len(results)} workers not wearing safty belts, where {stat}' 
        return nature_language



    def save_img(self,response,img_base64, path_file):
        img_cv = self.base64_to_cv2(img_base64)
        results = response["data"][0]
        for res in results:
            x1,y1,x2,y2 = res['boundingCoordinates'][0][0], res['boundingCoordinates'][0][1], res['boundingCoordinates'][2][0], res['boundingCoordinates'][2][1]
            # print(x1,y1,x2,y2,res["tag"])
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255),thickness=5)
            cv2.putText(img_cv, "%s:%.4f" % (res["tag"], res['confidence']), (int(x1) + 2, int(y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(path_file, img_cv)    
        return 