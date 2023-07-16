import requests


def image_preprocess(url, category_name, process_src: str, process_dst: str,
                     process_width=768, process_height=768, preprocess_txt_action="ignore",
                     process_flip=False, process_split=False, process_caption=False):
    print(f"Processing images of category {category_name}")
    payload = {
        "id_task":0,
        "process_src": process_src,
        "process_dst": process_dst,
        "process_width": process_width,
        "process_height": process_height,
        "preprocess_txt_action": preprocess_txt_action,
        "process_keep_original_size":True,
        "process_flip": process_flip,
        "process_split": process_split,
        "process_caption": process_caption
    }
    requests.post(url=f'{url}/sdapi/v1/preprocess', json=payload)
