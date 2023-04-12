# from pathlib import Path
import torch
import torch.nn as nn
import clip
import os.path as osp


force_cpu = False
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"

scoring_state_name = osp.join("./checkpoints", "mlp", "sac+logos+ava1-l14-linearMSE.pth")
scoring_pt_state = torch.load(scoring_state_name, map_location=torch.device(device=device))  # for mlp
scoring_clip_model, scoring_clip_preprocess = clip.load("ViT-L/14", device=device,
                                                        download_root=osp.join("./checkpoints", "clip"))
# cls_clip_model, cls_clip_preprocess = clip.load("ViT-B/32", device=device,
#                                                 download_root=osp.join("./checkpoints", "clip"))
cls_clip_model = scoring_clip_model
cls_clip_preprocess = scoring_clip_preprocess

scoring_clip_model.eval()
cls_clip_model.eval()


def get_clip_image_features(image, device=device, task='scoring'):
    if task == 'scoring':
        model = scoring_clip_model
        preprocess = scoring_clip_preprocess
    else:
        model = cls_clip_model
        preprocess = cls_clip_preprocess
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    # image_features = image_features.cpu().detach().numpy()
    return image_features


def get_norm_layer(norm_layer):
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "batch":
        return nn.BatchNorm1d
    return None


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU


class ClipAestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class ClipClassifier(nn.Module):
    def __init__(self, input_dim=768,
                 num_classes=2, layers=[4096, 2048],
                 activation="relu", norm_layer="batch") -> None:
        super().__init__()

        self.activation = get_activation(activation)
        self.norm_layer = get_norm_layer(norm_layer)
        self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])

        if self.norm_layer is not None:
            self.layers.append(self.norm_layer(layers[0]))

        if len(layers) > 1:
            for i in range(1, len(layers)):
                self.layers.append(nn.Linear(layers[i-1], layers[i]))

                if self.norm_layer is not None:
                    self.layers.append(self.norm_layer(layers[i]))

                self.layers.append(self.activation())

        self.layers.append(nn.Linear(layers[-1], num_classes, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


"""
def get_score(image):
    image_features = get_image_features(image)
    score = predictor(image_features.to(device).float())
    # score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

class clipscorer(object):
    def __init__(self):
        pass

    def __call__(self):
        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = ClipAestheticPredictor(768)
        predictor.load_state_dict(pt_state)  # 必须载入
        predictor.to(device)
        return predictor
"""
