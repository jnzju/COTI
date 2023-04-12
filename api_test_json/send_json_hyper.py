import requests
import json
import os.path as osp
import time

url = "http://127.0.0.1:7860"


def create_hypernetwork(category_name="axolotl_test", enable_sizes=[768, 320, 640, 1280], overwrite_old: bool = False,
                        layer_structure=[1, 2, 1], activation_func="relu", weight_init="Normal"):
    print(f"Create hypernetwork {category_name}.pt")
    expected_path = osp.join(osp.abspath('..'), 'stable-diffusion-webui', 'models', 'hypernetworks', f"{category_name}.pt")
    if osp.exists(expected_path) and not overwrite_old:
        print(f"Hypernetwork {category_name}.pt already exists! Skip this step...")
        return expected_path
    if overwrite_old:
        print(f"Overwriting the old hypernetwork!")
    else:
        print(f"Creating new hypernetwork {category_name}.pt ...")
    payload = {
        "name": category_name,
        "enable_sizes": enable_sizes,
        "overwrite_old": overwrite_old,
        "layer_structure": layer_structure,
        "activation_func": activation_func,
        "weight_init": weight_init
    }
    response = requests.post(url=f'{url}/sdapi/v1/create/hypernetwork', json=payload)
    path = response.json()['info'][30:]
    print(f"Hypernetwork created at: {path}")
    return path


def send_singal(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        payload = json.load(fp)
    for key, val in payload.items():
        print(f"{key}: {val}, {type(val)}")
    response = requests.post(url=f'{url}/sdapi/v1/train/hypernetwork', json=payload)
    # response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    print(response)
    print(response.json())


if __name__ == '__main__':
    create_hypernetwork("axolotl_test", overwrite_old=False)
    time1 = time.time()
    send_singal("signal_hyper_1.json")
    time2 = time.time()
    # send_singal("signal_hyper_2.json")
    # time3 = time.time()
    print(f"Phase 1: {time2 - time1}s")
    # print(f"Phase 2: {time3 - time2}s")
