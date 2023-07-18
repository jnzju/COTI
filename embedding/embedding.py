import requests
import os
import os.path as osp
import datetime


def create_embedding(url, sd_path, category_name, nvpt: int = 8, overwrite_old: bool = True):
    print(f"Create embedding {category_name}.pt")
    expected_path = osp.join(sd_path, 'embeddings', f"{category_name}.pt")
    if osp.exists(expected_path) and not overwrite_old:
        print(f"Embedding {category_name}.pt already exists! Skip this step...")
        return expected_path
    if osp.exists(expected_path) and overwrite_old:
        os.remove(expected_path)
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


def embedding_training(server, sd_path, url, category_name, learn_rate=5e-4, batch_size=2, gradient_step=1,
                       data_root=None,
                       log_directory=None,
                       training_width=768, training_height=768,
                       steps=1000, initial_step=0, shuffle_tags=False, tag_drop_out=False,
                       latent_sampling_method="once",
                       save_embedding_every=5,
                       template_filename=None,
                       preview_prompt="a_photo_of_axolotl, axolotl, real_life",
                       preview_negative_prompt="lowres, text, error, cropped, worst quality, low quality, "
                                               "normal quality, jpeg artifacts, signature, watermark, username, blurry",
                       preview_steps=50, preview_sampler_index=0, preview_cfg_scale=7,
                       preview_seed=-1, preview_width=768, preview_height=768):
    payload = {}
    if server == "gpu25":
        payload = {
            "id_task": 0,
            "embedding_name": category_name,
            "learn_rate": str(learn_rate),
            "batch_size": batch_size,
            "gradient_step": gradient_step,
            "data_root": data_root,
            "log_directory": log_directory,
            "training_width": training_width,
            "training_height": training_height,
            "varsize": False,
            "steps": steps,
            "clip_grad_mode": "value",
            "shuffle_tags": shuffle_tags,
            "clip_grad_value": str(learn_rate),
            "tag_drop_out": tag_drop_out,
            "latent_sampling_method": latent_sampling_method,
            "use_weight": False,
            "create_image_every": save_embedding_every,
            "save_embedding_every": save_embedding_every,
            "template_filename": template_filename,
            "save_image_with_stored_embedding": False,
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

        requests.post(url=f'{url}/sdapi/v1/train/embedding', json=payload)

        embedding_path_list = [osp.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), category_name, 'embeddings',
                                        f"{category_name}-{initial_step + save_embedding_every * (i + 1)}.pt")
                            for i in range((steps - initial_step) // save_embedding_every)]

        image_path_list = [osp.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), category_name, 'images',
                                    f"{category_name}-{initial_step + save_embedding_every * (i + 1)}.png")
                        for i in range((steps - initial_step) // save_embedding_every)]
    elif server == "gpu_school":
        template_file = os.path.join(sd_path, "textual_inversion_templates", template_filename)
        payload = {
            "embedding_name": category_name,
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
            "create_image_every": save_embedding_every,
            "save_embedding_every": save_embedding_every,
            "template_file": template_file,
            "save_image_with_stored_embedding": False,
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

        info = requests.post(url=f'{url}/sdapi/v1/train/embedding', json=payload)

        print("embedding info")
        print(info)

        embedding_path_list = [osp.join(log_directory, category_name, 'embeddings',
                                        f"{category_name}-{initial_step + save_embedding_every * (i + 1)}.pt")
                            for i in range((steps - initial_step) // save_embedding_every)]

        image_path_list = [osp.join(log_directory, category_name, 'images',
                                    f"{category_name}-{initial_step + save_embedding_every * (i + 1)}.png")
                        for i in range((steps - initial_step) // save_embedding_every)]
    
    else :
        raise NotImplementedError
    return embedding_path_list, image_path_list
