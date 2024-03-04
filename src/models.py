import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from typing import Literal, Optional, Union


from losses import ArcMarginProduct
from torchvision.models import VisionTransformer


class CustomVit(nn.Module):
    def __init__(
        self,
        backbone: Optional[Literal["vit_b_16"]] = "vit_b_16",
        image_size: int = 224,
        num_classes: int = 15,
        cell_embedding_dim: int = 12,
        **kwargs,
    ):
        # cell type expected to have 4 classes,
        """
        Uses ViT_B_16 (default: image_size:224, patch_size 16):
            patch_size: 16,
            num_layers:12,
            num_heads: 12,
            hidden_dim: 768,
            mlp_dim: 3072
        see table 1.
        Feeds in cell embedding with the final mlp
        """

        super().__init__()

        if backbone is None:
            model_backbone = VisionTransformer(
                image_size=image_size, num_classes=num_classes, **kwargs
            )
        else:
            model_fn = getattr(torchvision.models, backbone)
            if image_size == 224 or image_size is None:
                model_backbone = model_fn(pretrained=True, **kwargs)
            else:
                model_backbone = model_fn(image_size=image_size, **kwargs)

        self.hidden_dim = model_backbone.hidden_dim  # 768 for vit_b_16
        self.patch_size = model_backbone.patch_size

        assert (
            image_size % self.patch_size == 0
        ), f"Image size({image_size}) and patch size must be evenly divisible ({self.patch_size})"

        self.num_classes = num_classes
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.conv_proj = nn.Conv2d(
            6, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.encoder = model_backbone.encoder
        self.heads = nn.Linear(self.hidden_dim + cell_embedding_dim, self.num_classes)
        self.cell_embedding_dim = cell_embedding_dim
        if cell_embedding_dim > 0:
            self.cell_embedding = nn.Embedding(4, cell_embedding_dim)

        self.arc_margin_product = ArcMarginProduct(
            self.hidden_dim + cell_embedding_dim, self.num_classes
        )

    def forward(self, x, s):
        n, c, h, w = x.shape

        x = self.conv_proj(
            x
        )  # after conv: batches, hidden_dim, num_patches, num_patches
        x = x.flatten(-2, -1)  # flatten patches: batches, hidden_dim, num_patches**2
        x = x.permute(
            0, 2, 1
        )  # attention needs: batches, flattened patches, patches_embedding (aka hidden_dim)
        x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)

        # bs, 197, hidden_dim
        x = self.encoder(x)[:, 0]
        if self.cell_embedding_dim > 0:
            x = torch.cat([x, self.cell_embedding(s).squeeze(1)], dim=1)
        return self.arc_margin_product(x), self.heads(x)


densenet_model_types = Literal[
    "densenet121", "densenet169", "densenet201", "densenet161"
]
resnet_model_types = Literal["resnet34", "resnet50", "resnet101", "resnet152"]


class CustomDensenet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: Union[densenet_model_types, resnet_model_types] = "densenet121",
        embedding_size: int = 512,
        cell_embedding_dim: int = 12,
    ):
        super().__init__()
        # cell type expected to have 4 classes
        self.backbone = backbone
        if backbone.startswith("densenet"):
            self._setup_densenet()
        elif backbone.startswith("resnet"):
            self._setup_resnet()
        else:
            raise ValueError(f"Unrecognized densenet/resnet model type: {backbone}")

        self.classes = num_classes
        self.embedding_size = embedding_size
        self.cell_embedding_dim = cell_embedding_dim

        if self.cell_embedding_dim > 0:
            self.features_num += cell_embedding_dim
            self.cell_embedding = nn.Embedding(4, cell_embedding_dim)

        # neck
        self.neck = nn.Sequential(
            nn.BatchNorm1d(self.features_num),
            nn.Linear(self.features_num, self.embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.BatchNorm1d(self.embedding_size),
        )

        # 2 "heads"
        self.arc_margin_product = ArcMarginProduct(self.embedding_size, self.classes)
        self.classification = nn.Linear(self.embedding_size, self.classes)

    def embed(self, x, s):
        x = self.features(x)  # uses embedding backbone, (bs, 1024, 16,16)

        # maybe dontuse adaptive pooling here?
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (bs, 1024, 1, 1)
        x = x.view(x.size(0), -1)  # (bs, 1024)
        if self.cell_embedding_dim > 0:
            x = torch.cat(
                [
                    x,
                    self.cell_embedding(s).squeeze(1),  # extra dim createdh here? check
                ],
                dim=1,
            )

        embedding = self.neck(x)
        return embedding

    def forward(self, x, s):
        # return both the embedding & logits for classification
        embedding = self.embed(x, s)
        return self.arc_margin_product(embedding), self.classification(embedding)

    def _setup_densenet(
        self,
        backbone: Literal[
            "densenet121", "densenet169", "densenet201", "densenet161"
        ] = "densenet121",
    ):
        channels = 96 if backbone == "densenet161" else 64
        pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
        self.features = pretrained_backbone.features
        self.features.conv0 = nn.Conv2d(
            6, channels, 7, 2, 3, bias=False
        )  # use 6 channel conv
        self.features_num = pretrained_backbone.classifier.in_features

    def _setup_resnet(
        self,
        backbone: Literal[
            "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet34",
    ):
        pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
        pretrained_backbone.conv1 = nn.Conv2d(
            6, 64, 7, 2, 3, bias=False
        )  # replace 1 st conv layer

        self.features = nn.Sequential(*list(pretrained_backbone.children())[:-2])
        self.features_num = pretrained_backbone.fc.in_features
