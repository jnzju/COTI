import requests
import io
import base64
import os.path as osp
import os
from PIL import Image, PngImagePlugin


def generate_virtual_dataset(url: str, prompt: str, num_samples: int, temp_dir: str,
                             width: int = 768, height: int = 768, batch_size: int = 10,
                             negative_prompt: str = "lowres, bad_anatomy, text, error, worst_quality, "
                                                    "low_quality, normal_quality, jpeg_artifacts, signature, "
                                                    "watermark, username, blurry"):
    payload = {
        "prompt": prompt,
        "seed": -1,
        "batch_size": batch_size,
        "n_iter": num_samples // batch_size + 1,
        "steps": 50,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "negative_prompt": negative_prompt,
        "save_local_image": True}

    os.makedirs(temp_dir, mode=0o777, exist_ok=True)

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    generated_data_info = []
    for idx, i in enumerate(r['images']):
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(osp.join(temp_dir, f'output_{idx}.png'), pnginfo=pnginfo)
        generated_data_info.append({"img": osp.join(temp_dir, f'output_{idx}.png'), "gt_label": 1, "aesthetic_score": 0.0, 'type': 'val'})

    return generated_data_info
