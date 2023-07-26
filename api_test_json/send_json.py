import requests
import json
import os.path as osp
import time

url = "http://127.0.0.1:7861"


def create_embedding(url, category_name, nvpt: int = 8, overwrite_old: bool = False):
    print(f"Create embedding {category_name}.pt")
    expected_path = osp.join(osp.abspath('..'), 'stable-diffusion-webui', 'embeddings', f"{category_name}.pt")
    if osp.exists(expected_path) and not overwrite_old:
        print(f"Embedding {category_name}.pt already exists! Skip this step...")
        return expected_path
    if overwrite_old:
        print(f"Overwriting the old embedding!")
    else:
        print(f"Creating new embedding {category_name}.pt ...")
    payload = {
        "name": category_name,
        "num_vectors_per_token": nvpt,
        "overwrite_old": overwrite_old
    }
    response = requests.post(url=f'{url}/sdapi/v1/create/embedding', json=payload)
    path = response.json()['info'][27:]
    print(f"Embedding created at: {path}")
    return path


def send_singal(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        payload = json.load(fp)
    response = requests.post(url=f'{url}/sdapi/v1/preprocess', json=payload)
    # response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    print(response)
    print(response.text)


if __name__ == '__main__':
    # create_embedding(url, "axolotl_test", overwrite_old=False)
    time1 = time.time()
    send_singal("signal_1.json")
    time2 = time.time()
    # send_singal("signal_2.json")
    # time3 = time.time()
    print(f"Phase 1: {time2 - time1}s")
    # print(f"Phase 2: {time3 - time2}s")
