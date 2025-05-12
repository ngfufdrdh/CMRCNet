import torch
from torch import nn
import torch.nn.functional as F
from config import get_config
from model.modules import ProjectionHead
from model.modules import SNN_Block,  MultiheadAttention
from model.utils_loss_function import cross_entropy, ReconstructionLoss
from model.vit_model import vit_base_patch32_224_path_encoder,  Block_fusion_MCFN_2

args = get_config()


class CLIPModel_ViT_itm_v14_MSE(nn.Module):
    """
    """
    def __init__(
            self,
            temperature=args.temperature,
            image_embedding=768,
            spot_embedding=args.spot_embedding,
            mask_prob=0.5
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.image_encoder = vit_base_patch32_224_path_encoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=256)  # aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=256)  # 3467 shared hvgs
        self.temperature = temperature
        self.spot_projection_restruction = SNN_Block(dim1=spot_embedding, dim2=2048)
        self.latent_linear_768_2_256 = nn.Linear(768, 256)
        self.first_attention = MultiheadAttention(embed_dim=256, num_heads=4)
        Block_cross_fusion = Block_fusion_MCFN_2(dim=256, num_heads=8, mlp_ratio=4, qkv_bias=True, drop_ratio=0.1)
        self.cross_modal_image_layers = nn.ModuleList([Block_cross_fusion for _ in range(2)])


    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])  # [b, 50, 768]
        spot_features = batch["st_data"]
        #         spot_features = self.spot_encoder(batch["reduced_expression"])

        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = image_features.mean(dim=1)
        image_embeddings = self.image_projection(image_embeddings)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)

        spot_features = self.spot_projection_restruction(spot_features)
        b, feature_dim = spot_features.size()
        spot_features = spot_features.view(b, -1, 256)      # [b, 8, 256]

        image_features = self.latent_linear_768_2_256(image_features)
        b, n, c = image_features.size()
        mask = torch.rand(b, n, c).to(image_features.device) < self.mask_prob
        image_features_mask = image_features * mask.float()

        latent, _ = self.first_attention(image_features_mask, spot_features)
        latent = latent.permute(1, 0, 2)
        for image_layer in self.cross_modal_image_layers:
            latent, spot_features = image_layer(latent, spot_features)

        image_restruction = latent.mean(dim=1)
        loss_restruction = ReconstructionLoss()(image_restruction, image_embeddings)

        loss_contrastive = torch.log(1 + loss)
        loss_restruction = torch.log(1 + loss_restruction)

        return loss_contrastive.mean(), loss_restruction.mean()



if __name__ == '__main__':
    images = torch.randn(4, 3, 224, 224).cuda()
    input_ids = torch.randint(5, 300, size=(4, 25)).cuda()
    attention_mask = torch.ones(4, 25).cuda()
    reduced_expression = torch.randn(4, 3467).cuda()
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'st_data': reduced_expression
    }

    CLIP = CLIPModel_ViT_itm_v14_MSE().cuda()
    loss_1, loss_2 = CLIP(batch)
    print("")