import numpy as np
from scipy import linalg
from scipy.stats import entropy
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3

import torch
import pathlib
from PIL import Image
import os
import numpy as np
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from piq import FID,IS
import piq
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_fid.fid_score import calculate_fid_given_paths

class InceptionScore1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3().to(device=device)
        self.up = nn.Upsample(size=(299, 299), mode="bilinear").to(
            device=device
        )

    def forward(self, x):
        """x inputs should be (N, 3, 299, 299) in range -1 to 1.

        Returns class probabilities in form of torch.tensor of shape
        (N, 1000, 1, 1).
        """
        print('x')
        print(x.shape)
        x = self.up(x)
        x = self.inception_model(x)
        # print(x.shape)
        x = torch.softmax(x, dim=1)
        # x = torch.unsqueeze(x, 2)
        # x = torch.unsqueeze(x, 3)
        return x.data.cpu().numpy()


def calculate_inception_score(
    sample_dataloader,
    batch_size,
    device="cuda",
    num_images=50000,
    splits=10,
):
    """Calculate the inception score for a model's samples.

    Args:
        sample_dataloader: Dataloader for the generated image samples from the
            model.
        test_dataloader: Dataloader for the real images from the dataset to
            compare to.
        device: to perform the evaluation (e.g. 'cuda' for GPU).
        num_images: number of images to evaluate.
        splits: number of splits to perform for the evaluation.

    Returns:
        dict: Dictionary with key being the metric name, and values being the
            metric scores.
    """
    inception_model = InceptionScore(device=device)
    inception_model.eval()

    preds = np.zeros((num_images, 1000))

    for i, batch in enumerate(sample_dataloader, 0):
        batch = batch.to(device=device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        predictions = inception_model(batchv)
        start = i * batch_size
        n_predictions = len(preds[start : start + batch_size_i])
        preds[start : start + batch_size_i] = predictions[:n_predictions]

    split_scores = []

    for k in range(splits):
        part = preds[
            k * (num_images // splits) : (k + 1) * (num_images // splits), :
        ]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
        
    print(split_scores)

    return {"Inception Score": np.mean(split_scores)}



def evaluate_image_generation_gan(
    model, model_output_transform, test_loader, device="cuda"
):
    """Evaluate the image generation performance for a GAN.

    Evaluation will be against a base dataset (test_loader).

    Args:
        model: PyTorch model instance.
        model_output_transform: a function that transforms the model output
            (e.g. torch.Tensor output). This would be done, for instance, to
            normalize outputs to the right values (between -1 and 1 for
            inception).
        test_loader: The Dataloader for the test dataset (e.g. CIFAR-10).
        device: to perform the evaluation (e.g. 'cuda' for GPU).

    Returns:
        dict: Dictionary with keys as metric keys, and values as metric values.
    """

    num_images = 50000

    noise, _ = model.buildNoiseData(num_images)
    noise_dataloader = torch.utils.data.DataLoader(
        noise, batch_size=test_loader.batch_size
    )

    output = None
    with torch.no_grad():
        for i, noise_batch in enumerate(noise_dataloader):
            partial_output = model.test(noise_batch).to(device=device)
            if output is None:
                output = partial_output
            else:
                output = torch.cat((output, partial_output))

    if model_output_transform is not None:
        output = model_output_transform(
            output, target=None, device=device, model=model
        )

    # Set up dataloader
    sample_dataloader = torch.utils.data.DataLoader(
        output, batch_size=test_loader.batch_size
    )

    # Calculate Metrics
    inception_score = calculate_inception_score(
        test_dataloader=test_loader,
        sample_dataloader=sample_dataloader,
        device=device,
        num_images=num_images,
    )

    return {**inception_score, **fid}

class Images(Dataset):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.files = list(self.path.glob('*.png'))
        self.img_transform=transforms.Compose([
                            transforms.Resize(size=256),
                            transforms.RandomCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    def __getitem__(self, index):
        ori_img = Image.open(self.files[index])
        img = self.img_transform(ori_img)
        img = (img+1)/2
        # ori_img = preprocess(ori_img)
        return img

    def __len__(self):
        return len(self.files)
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
# from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.multimodal import CLIPScore
from clip_score.clip_score import main as calculate_clip_score
from clip_score.clip_score import parser
from clip_score.clip_score import DummyDataset
import argparse

if __name__ == '__main__':
    init_data_dir = '/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/stable_diffusion_dataset/training_dataset_initial/axolotl'
    
    gen_data_dir = '/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/tasks/2023-07-22-15-44_a0dca728/task_0/gen_val_images'
    text_dir = '/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/text'
    
    # parser.real_path = gen_data_dir
    # parser.fake_path = text_dir
    print(calculate_clip_score(real_path=gen_data_dir, fake_path=text_dir))
    
    # gen_loader = DataLoader(Images(gen_data_dir), batch_size=init_loader_size)
    
    # print(calculate_inception_score(gen_loader, init_loader, num_images=len(Images(gen_data_dir))))
    # print(calculate_fid(gen_loader, init_loader, num_images=len(Images(gen_data_dir))))
    
    
    
    # transforms=transforms.Compose([
    #                         transforms.Resize(size=299),
    #                         transforms.RandomCrop(size=299),
    #                         transforms.ToTensor()
    #                          ])
    
    # files_gen = list(pathlib.Path(gen_data_dir).glob('*.png'))
    # gen_tensor = torch.stack(
    #     ([transforms(Image.open(img_path)).to(torch.uint8) for img_path in files_gen])
    # )
    
    # metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    # score = metric(gen_tensor, "a photo of a axolotl")
    # print(score.detach())
    
    
    # files_init = list(pathlib.Path(init_data_dir).glob('*.png'))
    # init_tensor = torch.stack(
    #     ([transforms(Image.open(img_path)).to(torch.uint8) for img_path in files_init])
    # )
    
    # fid = FrechetInceptionDistance(feature=2048)
    # print(init_tensor.shape)
    # print(gen_tensor.shape)
    # fid.update(init_tensor, real=True)
    # fid.update(gen_tensor, real=False)
    # score = fid.compute()
    # print('fid')
    # print(score)
    
    
    # # print(gen_tensor.shape)
    # inception = InceptionScore()
    # inception.update(gen_tensor)
    # score = inception.compute()
    # print('is')
    # print(score)
    
    
    # image = transforms(Image.open(files[0]))
    # print('image')
    # print(image.shape)
    
    # dataset = ImagePathDataset(files, transforms=transforms)
    # print('dataset')
    # # print(dataset)
    # batch_size = len(os.listdir(gen_data_dir)) // 10
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          drop_last=False,
    #                                          num_workers=0
    #                                         )
    
    # score = calculate_inception_score(dataset, batch_size, 'cuda', num_images=len(os.listdir(gen_data_dir)), splits=10)
    # print(score)