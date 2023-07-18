import requests


def image_preprocess(server, url, category_name, process_src: str, process_dst: str,
                     process_width=768, process_height=768, preprocess_txt_action="ignore",
                     process_flip=False, process_split=False, process_caption=False):
    print(f"Processing images of category {category_name}")
    payload = {}
    if server == "gpu25":
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
    elif server == "gpu_school":
        payload = {
            "process_src": process_src,
            "process_dst": process_dst,
            "process_width": process_width,
            "process_height": process_height,
            "preprocess_txt_action": preprocess_txt_action,
            "process_flip": process_flip,
            "process_split": process_split,
            "process_caption": process_caption
        }
    else :
        raise NotImplementedError
    info = requests.post(url=f'{url}/sdapi/v1/preprocess', json=payload)
    print("preprocess info")
    print(info)
