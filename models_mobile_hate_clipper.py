import torch
import os
import torch.nn as nn
import mobileclip

class MobileHateClipper(nn.Module):
    def __init__(self, fusion='align', embed_dim = 1024, num_pre_output_layers=2, clip_model=None):
        super().__init__()

        # --------------------------------------------------------------
        # MHC encoding specifics
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
    
        self.embed_dim = embed_dim
        self.image_projection = torch.nn.Linear(512, embed_dim)
        self.image_dropout = torch.nn.Dropout(0.1)
        self.text_projection = torch.nn.Linear(512, embed_dim)
        self.text_dropout = torch.nn.Dropout(0.1)
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
            raise ValueError("fusion mode must be [align, concat, cross]")
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # MHC output specifics
        self.pre_dropout = torch.nn.Dropout(0.2)
        self.num_pre_output_layers = num_pre_output_layers
        self.pre_output_layers = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(0.2),
            nn.ReLU()
        ) for _ in range(self.num_pre_output_layers)])

        self.output_layer = torch.nn.Linear(embed_dim, 1)
        # --------------------------------------------------------------

    def forward(self, image, text):

        # encode image and text
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)

        # normalize features
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        # project features in FMI space
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        # dropout                                           
        image_features = self.image_dropout(image_features)                                
        text_features = self.text_dropout(text_features)

        image_features = nn.ReLU()(image_features)
        text_features = nn.ReLU()(text_features)

        # FMI fusion
        if self.fusion == "align":
            features = torch.mul(image_features, text_features) # [N, d]
        elif self.fusion == "cross":
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [N, d, d]
        else:
            raise ValueError("Invalid fusion method")
        
        # pre-output layers
        features = self.pre_dropout(features)
        for layer in self.pre_output_layers:
            features = layer(features)

        logits = self.output_layer(features)

        return logits
    


def create_model(args):
    checkpoint_path = os.path.join(args.clip_checkpoint, f'{args.clip_model}.pt')
    clip_model, _, _ = mobileclip.create_model_and_transforms(args.clip_model, pretrained=checkpoint_path)
    return MobileHateClipper(
        fusion=args.fusion,
        embed_dim=args.embed_dim,
        num_pre_output_layers=args.num_pre_output_layers,
        clip_model=clip_model
    )
