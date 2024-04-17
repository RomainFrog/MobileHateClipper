import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import mobileclip

class MobileHateClipper(nn.Module):
    def __init__(self, fusion='align', embed_dim = 1024, pre_output_dim=512,
                num_mapping_layers = 1, num_pre_output_layers=2, clip_model=None,
                dropout_rates=[0.1, 0.4, 0.2], freeze_clip=True):
        super().__init__()

        # --------------------------------------------------------------
        # MHC encoding specifics
        self.clip = clip_model
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        ## Create a mapping from the CLIP embeddings to the transformer input
        self.mapping = nn.Sequential(
            nn.Linear(embed_dim, pre_output_dim),
            nn.LayerNorm(pre_output_dim),
            nn.ReLU()
        )

        ## Create a single transformer layer that takes as input the CLIP embeddings and outputs the predicted logits
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=pre_output_dim, dropout=dropout_rates[0])


        ## Create a mapping from the transformer output to the logits
        self.pre_output = nn.Sequential(
            nn.Linear(pre_output_dim, pre_output_dim),
            nn.LayerNorm(pre_output_dim),
            nn.ReLU()
        )

        ## Create a final linear layer to output the logits
        self.output = nn.Linear(pre_output_dim, 2)
    
        

    def forward(self, image, text):

        # encode image and text
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)

        # map the CLIP embeddings to the transformer input
        image_features = self.mapping(image_features)
        text_features = self.mapping(text_features)

        # concatenate the image and text features
        features = torch.mul(image_features, text_features)

        # pass the concatenated features through the transformer
        features = self.transformer(features)

        # pass the transformer output through the pre-output layers
        features = self.pre_output(features)

        # pass the pre-output through the final linear layer to get the logits
        logits = self.output(features)

        return logits
    


def create_model(args):
    checkpoint_path = os.path.join(args.clip_checkpoint, f'{args.clip_model}.pt')
    clip_model, _, _ = mobileclip.create_model_and_transforms(args.clip_model, pretrained=checkpoint_path)
    return MobileHateClipper(
        fusion=args.fusion,
        embed_dim=args.embed_dim,
        pre_output_dim=args.pre_output_dim,
        num_pre_output_layers=args.num_pre_output_layers,
        clip_model=clip_model,
        dropout_rates=args.dropouts,
        freeze_clip=args.freeze_clip)
