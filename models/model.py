import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PointNetfeat, TextEncoder, PromptLearner, CrossAttention, feature_transform_regularizer
from models.lora import create_lora_image_encoder, create_lora_text_encoder
from utils import load_gpt_descriptions, Realistic_Projection
from utils.peft_utils import create_bitfit_image_encoder, create_ln_only_image_encoder, create_bitfit_text_encoder, create_ln_only_text_encoder
from clip import clip
import numpy as np

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def insert_prompts(self, x, prompt_embeddings):
        prompt_embeddings = prompt_embeddings.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([
            x[:,0,:].unsqueeze(1),
            prompt_embeddings,
            x[:,5:, :]
        ], dim = 1)
        return x
    
   

    def pre(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        return x
    
    def post(self, x):
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # insert prompts here
       
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class ImagePrompt(nn.Module):
    def __init__(self):
        super().__init__()
        image_ctx_vectors = torch.empty(4, 768)
        nn.init.normal_(image_ctx_vectors, std=0.02)
        self.image_ctx = nn.Parameter(image_ctx_vectors) 

    def forward(self):
        return self.image_ctx

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(2) * logits.log_softmax(2)).sum(2)
    
    idx_batch = []
    logits_chosen = []
    for i in range(batch_entropy.shape[0]):
        # Append indices with batch entropy below 0.5
        threshold = torch.quantile(batch_entropy[i], 0.5)
        idx_below_threshold = torch.where(batch_entropy[i] <= threshold)[0]
        idx_batch.append(idx_below_threshold)
        logits_chosen.append(logits[i, idx_batch[i], :])
    # logits_chosen = torch.stack(logits_chosen)
    # idx = torch.stack(idx_batch)

    # idx = torch.argsort(batch_entropy, descending=False)[:, :int(batch_entropy.size()[0] * top)]
    
    return logits_chosen, idx_batch

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    # out = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    # print(avg_logits)
    # print(out)
    # exit()
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class Model(nn.Module):
    def __init__(self, cfg, classnames, clip_model, peft_method='lora', text_peft_method='lora'):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.peft_method = peft_method
        self.text_peft_method = text_peft_method
        
        # self.point_encoder = NaivePCT()
        self.point_encoder = PointNetfeat(global_feat=True, feature_transform=True)
        # self.pc_mlp = nn.Linear(1024, 768) # relu Linear 1024 -> 256 -> 512
        self.pc_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True)
        )

        # Create parameter-efficient fine-tuning wrapped ImageEncoder
        original_image_encoder = ImageEncoder(clip_model.visual)
        
        if peft_method == 'lora':
            self.image_encoder = create_lora_image_encoder(
                original_image_encoder,
                rank=16,                           # Low-rank bottleneck dimension
                alpha=32,                          # LoRA scaling factor
                dropout=0.1,                       # Dropout for LoRA layers
                target_modules=['attn', 'mlp']     # Apply LoRA to attention and MLP layers
            )
        elif peft_method == 'bitfit':
            self.image_encoder = create_bitfit_image_encoder(
                original_image_encoder,
                vision_start=0                     # Start from first layer
            )
        elif peft_method == 'ln_only':
            self.image_encoder = create_ln_only_image_encoder(
                original_image_encoder,
                vision_start=0                     # Start from first layer
            )
        else:
            # No PEFT - use original encoder
            self.image_encoder = original_image_encoder

        # Create text encoder with optional PEFT
        original_text_encoder = TextEncoder(clip_model)

        if text_peft_method == 'lora':
            self.text_encoder = create_lora_text_encoder(
                original_text_encoder,
                rank=16,                           # Low-rank bottleneck dimension
                alpha=32,                          # LoRA scaling factor
                dropout=0.1,                       # Dropout for LoRA layers
                target_modules=['attn', 'mlp']     # Apply LoRA to attention and MLP layers
            )
        elif text_peft_method == 'bitfit':
            self.text_encoder = create_bitfit_text_encoder(
                original_text_encoder,
                text_start=0                       # Start from first layer
            )
        elif text_peft_method == 'ln_only':
            self.text_encoder = create_ln_only_text_encoder(
                original_text_encoder,
                text_start=0                       # Start from first layer
            )
        else:
            # No PEFT - use original encoder
            self.text_encoder = original_text_encoder
            
        self.token_embedding = clip_model.token_embedding

        self.cross_attention = CrossAttention(latent_dim = 512, kv_dim = 512, cross_heads = 8)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)  # ctx_init: a point could model of a
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = clip_model.logit_scale
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img
        self.num_views = 10
        self.channel = 512
        self.load_descriptions(clip_model, cfg)

        self.image_cross_attention = CrossAttention(latent_dim = 768, kv_dim = 512, cross_heads = 8)

        self.pc_mlp_2 = nn.Linear(768, 768)
        self.image_prompt_learner = ImagePrompt()


    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc.float()).to(pc.device)
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img

    
    def load_descriptions(self, clip_model, cfg):
        description_fname = 'PointDA_data/captions.json' if cfg.DATASET.NAME == 'PointDA' else 'GraspNetPointClouds/captions.json'
        hparams = {
            'descriptor_fname':description_fname,
            'category_name_inclusion': 'prepend',
            'before_text': '“the depth map of a ',
            'between_text': ' 3D model which has ',
            'after_text': '',
            'apply_descriptor_modification': None
        }
        descriptions, _ = load_gpt_descriptions(hparams)
        description_encodings = torch.stack([clip_model.encode_text(clip.tokenize(descriptions[clas])) for clas in descriptions])

        self.description_encodings = nn.Parameter(description_encodings.mean(dim=1).float(), requires_grad=False)

    
    def forward(self, point_cloud, label=None, use_pseudo_labels=False, centers=None):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): A tensor of shape (B, 30, 3, 224, 224) representing a batch of images.
            point_cloud (torch.Tensor): A tensor of shape (B, N, 3) representing a batch of point clouds.

        Returns:
            torch.Tensor: The output of the model.
        """
        B = point_cloud.size(0)
        logit_scale = self.logit_scale.exp()
        
        point_features, trans, trans_feat = self.point_encoder(point_cloud.permute(0, 2, 1))
        point_features = self.pc_mlp(point_features)

        image_prompt = self.image_prompt_learner()
        pc_condtioned_image_prompt = self.pc_mlp_2(self.image_cross_attention(data = point_features.unsqueeze(1), soft_prompt = image_prompt.unsqueeze(0).repeat(B, 1, 1)))
        with torch.no_grad():
            images = self.real_proj(point_cloud)
            images = images.reshape(B, 10, images.shape[-3], images.shape[-2], images.shape[-1])
        
        image_features_list = []
        for i in range(B):
            # with torch.no_grad():
            image_feat_i = self.image_encoder.pre(images[i])
            
            image_feat_i = self.image_encoder.insert_prompts(image_feat_i, pc_condtioned_image_prompt[i])

            # with torch.no_grad():
            image_feat_i = self.image_encoder.post(image_feat_i)

            image_features_list.append(image_feat_i)
            
      
        # Concatenate and normalize
        image_features = torch.cat(image_features_list, dim=0)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if not self.cfg.TRAINER.USE_CONFIDENCE_SAMPLING:
            image_features = image_features.reshape(B, self.num_views, self.channel).mean(dim=1)
        else:
            image_features = image_features.reshape(B, self.num_views, self.channel)

        
        if use_pseudo_labels and centers is not None:
            label = torch.argmax(image_features.mean(dim=1) @ centers.to(image_features.device).t(), dim=-1)
            

        ctx = self.prompt_learner.ctx
        description_encodings = self.description_encodings

        ctx_attended = self.cross_attention(data = description_encodings, soft_prompt = ctx)

        ctx_attended = ctx_attended.unsqueeze(0)

        prompt_with_point_features = self.prompt_learner(ctx_attended.unsqueeze(1).repeat(B, self.n_cls, 1, 1))
        
        logits = []
        for pts_i, imf_i in zip(prompt_with_point_features, image_features):
            text_features = self.text_encoder(pts_i, self.tokenized_prompts).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)

        # align_loss = F.mse_loss(image_features, point_features.squeeze(1))
        align_loss = torch.tensor(0)

        if self.cfg.TRAINER.USE_CONFIDENCE_SAMPLING:
            output, idx_selected = select_confident_samples(logits, 0.5)
            output_mean = torch.stack([i.mean(dim=0) for i in output])
            entropy_loss = []
            for i in range(len(output)):
                entropy_loss.append(avg_entropy(output[i]))
            entropy_loss = torch.mean(torch.stack(entropy_loss))
        else:
            output_mean, idx_selected = logits, None
            entropy_loss = torch.tensor(0)

        if label is not None:
            return output_mean, F.cross_entropy(output_mean, label), feature_transform_regularizer(trans_feat) if trans_feat is not None else torch.tensor(0), align_loss, entropy_loss
        else:
            return output_mean, feature_transform_regularizer(trans_feat) if trans_feat is not None else torch.tensor(0), align_loss, entropy_loss
    
    @torch.no_grad()
    def get_image_features(self, point_cloud):
        """
        Extract image features from a batch of point clouds.
        
        Args:
            point_cloud: Point cloud batch (B, N, 3)
            
        Returns:
            torch.Tensor: Image features (B, feature_dim)
        """
        B = point_cloud.size(0)
        
        # Extract point features for cross-attention
        point_features, _, _ = self.point_encoder(point_cloud.permute(0, 2, 1))
        point_features = self.pc_mlp(point_features)

        # Get image prompt conditioned on point cloud
        image_prompt = self.image_prompt_learner()
        pc_condtioned_image_prompt = self.pc_mlp_2(self.image_cross_attention(data = point_features.unsqueeze(1), soft_prompt = image_prompt.unsqueeze(0).repeat(B, 1, 1)))
        
        # Generate images from point clouds
        with torch.no_grad():
            images = self.real_proj(point_cloud)
            images = images.reshape(B, 10, images.shape[-3], images.shape[-2], images.shape[-1])
        
        # Extract image features
        image_features_list = []
        for i in range(B):
            image_feat_i = self.image_encoder.pre(images[i])
            image_feat_i = self.image_encoder.insert_prompts(image_feat_i, pc_condtioned_image_prompt[i])
            image_feat_i = self.image_encoder.post(image_feat_i)
            image_features_list.append(image_feat_i)
            
        # Concatenate and normalize
        image_features = torch.cat(image_features_list, dim=0)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Aggregate features across views
        image_features = image_features.reshape(B, self.num_views, self.channel).mean(dim=1)
        
        return image_features

    def get_peft_parameters(self):
        """
        Get PEFT-specific parameters based on the fine-tuning method used.
        Follows the same pattern as get_lora_parameters() for trainer integration.
        
        Returns:
            Dict mapping encoder names to lists of trainable parameters for PEFT methods
        """
        peft_params = {}
        
        # Get image encoder PEFT parameters
        if self.peft_method == 'lora' and hasattr(self.image_encoder, 'get_lora_parameters'):
            peft_params['image_encoder'] = self.image_encoder.get_lora_parameters()
        elif self.peft_method == 'bitfit' and hasattr(self.image_encoder, 'get_bitfit_parameters'):
            peft_params['image_encoder'] = self.image_encoder.get_bitfit_parameters()
        elif self.peft_method == 'ln_only' and hasattr(self.image_encoder, 'get_ln_only_parameters'):
            peft_params['image_encoder'] = self.image_encoder.get_ln_only_parameters()

        # Get text encoder PEFT parameters
        if self.text_peft_method == 'lora' and hasattr(self.text_encoder, 'get_lora_parameters'):
            peft_params['text_encoder'] = self.text_encoder.get_lora_parameters()
        elif self.text_peft_method == 'bitfit' and hasattr(self.text_encoder, 'get_bitfit_parameters'):
            peft_params['text_encoder'] = self.text_encoder.get_bitfit_parameters()
        elif self.text_peft_method == 'ln_only' and hasattr(self.text_encoder, 'get_ln_only_parameters'):
            peft_params['text_encoder'] = self.text_encoder.get_ln_only_parameters()
            
        return peft_params
    
    def get_lora_parameters(self):
        """
        Backward compatibility method for existing trainer code.
        Returns LoRA parameters if using LoRA, otherwise returns image encoder PEFT parameters for compatibility.
        """
        if self.peft_method == 'lora':
            peft_params = self.get_peft_parameters()
            return peft_params.get('image_encoder', [])
        else:
            # For non-LoRA methods, return image encoder PEFT parameters for compatibility
            peft_params = self.get_peft_parameters()
            return peft_params.get('image_encoder', [])


