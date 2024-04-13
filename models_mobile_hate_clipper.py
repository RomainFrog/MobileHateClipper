import torch
import os
import torch.nn as nn
import mobileclip

class MobileHateClipper(nn.Module):
    def __init__(self, fusion='align', num_pre_output_layers=2, clip_model=None):
        super().__init__()

        # --------------------------------------------------------------
        # MHC encoding specifics
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        self.image_projection = torch.nn.Linear(512, 512)
        self.text_projection = torch.nn.Linear(512, 512)
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # MHC Feature Interaction Matrix specifics
        self.fusion = fusion
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # MHC output specifics
        self.num_pre_output_layers = num_pre_output_layers
        self.pre_output_layers = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        ) for _ in range(self.num_pre_output_layers)])

        self.output_layer = torch.nn.Linear(512, 1)
        # --------------------------------------------------------------

    def forward(self, image, text):

        # encode image and text
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)

        # normalize features
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1, keepdim=True)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1, keepdim=True)

        # project features in FMI space
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        # FMI fusion
        if self.fusion == "align":
            features = torch.mul(image_features, text_features)
        elif self.fusion == "cross":
            features = torch.bmm(image_features, text_features.transpose(1, 2))
        else:
            raise ValueError("Invalid fusion method")
        
        # pre-output layers
        for layer in self.pre_output_layers:
            features = layer(features)

        output = self.output_layer(features)

        return output
    


def create_model(args):
    checkpoint_path = os.path.join(args.clip_checkpoint, f'{args.clip_model}.pt')
    clip_model, _, _ = mobileclip.create_model_and_transforms(args.clip_model, pretrained=checkpoint_path)
    return MobileHateClipper(
        fusion=args.fusion,
        num_pre_output_layers=args.num_pre_output_layers,
        clip_model=clip_model
    )