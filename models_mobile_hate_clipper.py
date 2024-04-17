import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import mobileclip
from transformers import CLIPModel


class MobileHateClipper(nn.Module):
    def __init__(self, fusion='align', embed_dim = 1024, encoder_dim=512, pre_output_dim=512,
                num_mapping_layers = 1, num_pre_output_layers=2, clip_model=None,
                dropout_rates=[0.1, 0.4, 0.2], freeze_clip=True):
        super().__init__()

        # --------------------------------------------------------------
        # MHC encoding specifics
        self.clip = clip_model
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
    
        self.embed_dim = embed_dim
        self.num_mapping_layers = num_mapping_layers
        image_mapping_layers = [torch.nn.Linear(encoder_dim, self.embed_dim), nn.Dropout(p=dropout_rates[0])]
        text_mapping_layers = [torch.nn.Linear(encoder_dim, self.embed_dim), nn.Dropout(p=dropout_rates[0])]
        for _ in range(1, num_mapping_layers):
            image_mapping_layers.extend([nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim), nn.Dropout(p=dropout_rates[0])])
            text_mapping_layers.extend([nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim), nn.Dropout(p=dropout_rates[0])])

        self.image_projection = nn.Sequential(*image_mapping_layers)
        self.text_projection = nn.Sequential(*text_mapping_layers)

        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # MHC Feature Interaction Matrix specifics
        self.fusion = fusion
        if self.fusion == 'align':
            self.pre_output_input_dim = self.embed_dim
        elif self.fusion == 'concat':
            self.pre_output_input_dim = self.embed_dim * 2
        elif self.fusion == 'cross':
            self.pre_output_input_dim = self.embed_dim ** 2
        else:
            raise ValueError("fusion mode must be in [align, concat, cross]")
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # MHC output specifics

        self.num_pre_output_layers = num_pre_output_layers
        pre_output_layers = [nn.Dropout(p=dropout_rates[1])]
        pre_output_layers.extend([nn.Linear(self.pre_output_input_dim, pre_output_dim), nn.ReLU(), nn.Dropout(p=dropout_rates[2])])
        for _ in range(1, num_pre_output_layers):
            pre_output_layers.extend([nn.Linear(pre_output_dim, pre_output_dim), nn.ReLU(), nn.Dropout(p=dropout_rates[2])])

        self.pre_output_layers = nn.Sequential(*pre_output_layers)

        self.output_layer = nn.Linear(pre_output_dim, 1)
        # --------------------------------------------------------------

    def forward(self, image, text):

        # encode image and text
        image_features = self.clip.vision_model(image).pooler_output
        text_features = self.clip.text_model(text).pooler_output
        image_features = self.clip.visual_projection(image_features)
        text_features = self.clip.text_projection(text_features)

        print(f'[INFO] {image_features.shape=}')
        print(f'[INFO] {text_features.shape=}')

        # project features
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        # normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)


        # FMI fusion
        if self.fusion == "align":
            features = torch.mul(image_features, text_features) # [N, d]
        elif self.fusion == "concat":
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion == "cross":
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [N, d, d]
            features = features.view(-1, self.embed_dim ** 2)
        else:
            raise ValueError("Invalid fusion method")
        
        # pre-output layers
        features = self.pre_output_layers(features)
        logits = self.output_layer(features)

        return logits
    


def create_model(args):
    clip_model = None
    if args.clip_model.startswith('mobileclip'):
        checkpoint_path = os.path.join(args.clip_checkpoint, f'{args.clip_model}.pt')
        clip_model, _, _ = mobileclip.create_model_and_transforms(args.clip_model, pretrained=checkpoint_path)
    else:
        clip_model = CLIPModel.from_pretrained(args.clip_model)

    return MobileHateClipper(
        fusion=args.fusion,
        embed_dim=args.embed_dim,
        pre_output_dim=args.pre_output_dim,
        num_pre_output_layers=args.num_pre_output_layers,
        clip_model=clip_model,
        dropout_rates=args.dropouts,
        freeze_clip=args.freeze_clip)
