import json

from PIL import Image, ImageOps
import numpy as np
import time
import torch
import requests
import io


# 通过 URL 获取图片
def get_image_from_url(url):
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        raise Exception(response.text)

    i = Image.open(io.BytesIO(response.content))

    i = ImageOps.exif_transpose(i)

    if i.mode != "RGBA":
        i = i.convert("RGBA")

    # recreate image to fix weird RGB image
    alpha = i.split()[-1]
    image = Image.new("RGB", i.size, (0, 0, 0))
    image.paste(i, mask=alpha)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    return image, mask


def send_request(input, parameters, api_key):
    data = json.dumps({
        "model": "wanx-anytext-v1",
        "input": input,
        "parameters": parameters
    })
    # 发送请求
    print('Sending request:data', data)
    # cURL
    # POST https://dashscope.aliyuncs.com/api/v1/services/aigc/anytext/generation
    try:
        response = requests.post(
            url="https://dashscope.aliyuncs.com/api/v1/services/aigc/anytext/generation",
            headers={
                "X-Dashscope-Async": "enable",
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            data=data
        )
        if response.status_code == 200:
            print('Request was successful')
            content = json.loads(response.content)
            # 如果HTTP响应状态码为200，则表示请求成功 执行下一步, 从 response.content 拿出来 output 的 task_id 作为下一步的输入
            print(content)
            task_id = content['output']['task_id']
            print(task_id)
            return task_id
        else:
            print('Response HTTP Status Code: {status_code}'.format(
                status_code=response.status_code))
            print('Response HTTP Response Body: {content}'.format(
                content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')


class AnyTextAPI_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mask_image_url": ("STRING", {"default": ""}),
                "base_image_url": ("STRING", {"default": ""}),
                "appended_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "layout_priority": (["vertical", "horizontal"], {"default": "vertical"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 99999999}),
                "api_key": ("STRING", {"default": "sk-"}),
                "image_width": ("INT", {"default": 512, "min": 64, "max": 7000}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": 7000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "AnyTextAPI_Node"
    FUNCTION = "execute"

    def execute(self, prompt, mask_image_url, base_image_url, appended_prompt, negative_prompt, layout_priority, steps,
                seed, api_key, image_width, image_height):
        print('Requesting image generation')
        params = {
            "input": {

            },
            "parameters": {

            },
            "api_key": api_key
        }
        if prompt:
            params["input"]["prompt"] = prompt
        if base_image_url:
            params["input"]["base_image_url"] = base_image_url
        if mask_image_url:
            params["input"]["mask_image_url"] = mask_image_url
        if appended_prompt:
            params["input"]["appended_prompt"] = appended_prompt
        if negative_prompt:
            params["input"]["negative_prompt"] = negative_prompt
        if layout_priority:
            params["parameters"]["layout_priority"] = layout_priority
        if steps:
            params["parameters"]["steps"] = steps
        if seed:
            params["parameters"]["seed"] = seed

        print('Looping request', params)
        api_key = params['api_key']
        task_id = send_request(params['input'], params['parameters'], api_key)
        loop = True
        max_retries = 100
        while loop and max_retries > 0:
            try:
                response = requests.get(
                    url="https://dashscope.aliyuncs.com/api/v1/tasks/" + task_id,
                    headers={
                        "X-Dashscope-Async": "enable",
                        "Authorization": "Bearer " + api_key,
                        "Content-Type": "application/json",
                    },
                )
                # 从 response.content 拿出来 output 的 task_status 和 result_url
                content = json.loads(response.content)
                print(content)
                task_status = content['output']['task_status']
                if task_status == 'SUCCEEDED':
                    print('Task is SUCCEEDED')
                    result_url = content['output']['result_url'][0]
                    print('Result URL: ' + result_url)
                    loop = False
                    return get_image_from_url(result_url)
                # 如果 task_status 是 FAILED, 则打印错误信息并退出
                elif task_status == 'FAILED':
                    print('Task is FAILED')
                    print('Error: ' + content['output']['message'])
                    loop = False
                else:
                    print('Task is still running')
                # 延迟1秒
                time.sleep(1)
                max_retries -= 1
                if max_retries == 0:
                    print('Max retries reached')
                    return None
            except requests.exceptions.RequestException:
                print('HTTP Request failed')


# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyTextAPI_Node": AnyTextAPI_Node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTextAPI_Node": "AnyTextAPI",
}
