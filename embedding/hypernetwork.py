import requests
import os
import os.path as osp


def create_hypernetwork(url, category_name, enable_sizes=[768, 320, 640, 1280], overwrite_old: bool = True,
                        layer_structure=[1, 2, 1], activation_func="relu", weight_init="Normal"):
    print(f"Create hypernetwork {category_name}.pt")
    expected_path = osp.join(osp.abspath('..'), 'stable-diffusion-webui', 'models', 'hypernetworks', f"{category_name}.pt")
    if osp.exists(expected_path) and not overwrite_old:
        print(f"Hypernetwork {category_name}.pt already exists! Skip this step...")
        return expected_path
    if osp.exists(expected_path) and overwrite_old:
        os.remove(expected_path)
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


def hypernetwork_training(url, category_name, learn_rate=5e-4, batch_size=1, gradient_step=1,
                          data_root="/home/yangjn/data/stable_diffusion_dataset/~Temp_selected_images/processed",
                          log_directory="/home/yangjn/temp_out",
                          training_width=768, training_height=768,
                          steps=1000, initial_step=0, shuffle_tags=False, tag_drop_out=False,
                          latent_sampling_method="once",
                          save_hypernetwork_every=50,
                          template_file="/home/yangjn/stable-diffusion-webui/textual_inversion_templates/hypernetwork.txt",
                          preview_prompt="a_photo_of_axolotl, axolotl, real_life",
                          preview_negative_prompt="lowres, text, error, cropped, worst quality, low quality, "
                                                  "normal quality, jpeg artifacts, signature, watermark, username, blurry",
                          preview_steps=50, preview_sampler_index=0, preview_cfg_scale=7,
                          preview_seed=-1, preview_width=768, preview_height=768):

    payload = {
        "hypernetwork_name": category_name,
        "learn_rate": str(learn_rate),
        "batch_size": batch_size,
        "gradient_step": gradient_step,
        "data_root": data_root,
        "log_directory": log_directory,
        "training_width": training_width,
        "training_height": training_height,
        "steps": steps,
        "shuffle_tags": shuffle_tags,
        "tag_drop_out": tag_drop_out,
        "latent_sampling_method": latent_sampling_method,
        "create_image_every": save_hypernetwork_every,
        "save_hypernetwork_every": save_hypernetwork_every,
        "template_file": template_file,
        "preview_from_txt2img": True,
        "preview_prompt": preview_prompt,
        "preview_negative_prompt": preview_negative_prompt,
        "preview_steps": preview_steps,
        "preview_sampler_index": preview_sampler_index,
        "preview_cfg_scale": preview_cfg_scale,
        "preview_seed": preview_seed,
        "preview_width": preview_width,
        "preview_height": preview_height
    }

    requests.post(url=f'{url}/sdapi/v1/train/hypernetwork', json=payload)

    hypernetwork_path_list = [osp.join(log_directory, category_name, 'hypernetworks',
                                       f"{category_name}-{save_hypernetwork_every * (i + 1)}.pt")
                              for i in range((steps - initial_step) // save_hypernetwork_every)]

    image_path_list = [osp.join(log_directory, category_name, 'images',
                                f"{category_name}-{save_hypernetwork_every * (i + 1)}.png")
                       for i in range((steps - initial_step) // save_hypernetwork_every)]

    return hypernetwork_path_list, image_path_list
