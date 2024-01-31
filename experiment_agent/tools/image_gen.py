import urllib.request
import cv2
import numpy as np 
import urllib.parse , json5
class Image_gen:

    def __call__(self,query, path_file):
        prompt = json5.loads(query)["prompt"]
        prompt = urllib.parse.quote(prompt)
        image_url = f'https://image.pollinations.ai/prompt/{prompt}'
        try : 
            self.save_img(image_url,path_file, prompt)
        except:
            pass    
        answer = json5.dumps({'image_url': image_url}, ensure_ascii=False)
        answer = f'我已经完成图片生成任务,图片网址为{answer}.'
        return answer
    
    def download_cv_image(self, img_url):
        req = urllib.request.Request(
                            img_url,
                            data=None,
                            headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                                        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                            }
                            )
        with urllib.request.urlopen(req) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image_as_cvimage = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image_as_cvimage  


    def save_img(self,image_url,path_file,prompt):
        img_cv = self.download_cv_image(image_url)
        out_file = f'{path_file}/show_{prompt}.jpg'
        cv2.imwrite(out_file, img_cv)    
        return 
    
if  __name__ == '__main__':
    image_ = Image_gen()
    image_.run('dog','./image.jpg')