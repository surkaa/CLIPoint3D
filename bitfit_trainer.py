import os.path as osp
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.samplers import build_sampler
from dassl.data.data_manager import build_data_loader
from models import load_clip_to_cpu, Model, feature_transform_regularizer
from utils import PointDA, EntropyLoss, Scannet, Wasserstein1Loss, GraspNet
from tqdm import tqdm
from geomloss import SamplesLoss
import time
import datetime
from dassl.utils import MetricMeter, AverageMeter

@TRAINER_REGISTRY.register()
class BitFitTrainer(TrainerXU):

    def build_model(self):
        cfg = self.cfg
        classnames = list(self.lab2cname.values())

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MODEL.PREC == "fp32" or cfg.TRAINER.MODEL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        print("Building MODEL with BitFit fine-tuning")
        self.model = Model(cfg, classnames, clip_model, peft_method='lora', text_peft_method='bitfit')
        print("Turning off gradients in both the image and the text encoder")

        name_to_update = ["prompt_learner", "point_encoder", "pc_mlp", "cross_attention", "image_prompt_mlp"]

        if hasattr(self.model, "view_weights") and cfg.TRAINER.USE_VIEW_WEIGHTS:
            name_to_update.append("view_weights") 
        # Check if image_encoder has BitFit parameters - if so, we'll optimize only those
        has_bitfit_image_encoder = (hasattr(self.model, 'image_encoder') and 
                                   hasattr(self.model.image_encoder, 'get_bitfit_parameters') and
                                   len(self.model.image_encoder.get_bitfit_parameters()) > 0)
        if has_bitfit_image_encoder:
            name_to_update.append("image_encoder")
            print("Image encoder has BitFit parameters - will optimize only BitFit (bias) parameters, not the full encoder.")

        # Check if text_encoder has BitFit parameters - if so, we'll optimize only those
        has_bitfit_text_encoder = (hasattr(self.model, 'text_encoder') and 
                                  hasattr(self.model.text_encoder, 'get_bitfit_parameters') and
                                  len(self.model.text_encoder.get_bitfit_parameters()) > 0)
        if has_bitfit_text_encoder:
            name_to_update.append("text_encoder")
            print("Text encoder has BitFit parameters - will optimize only BitFit (bias) parameters, not the full encoder.")

        # First freeze all parameters
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        # Then enable parameters in modules we want to train
        for name, param in self.model.named_parameters():
            if any(name_to_check in name for name_to_check in name_to_update):
                param.requires_grad_(True)
        
        # Special handling: For BitFit image_encoder, freeze non-bias params to save memory
        if has_bitfit_image_encoder:
            bitfit_params = self.model.image_encoder.get_bitfit_parameters()
            print(f"Found {len(bitfit_params)} BitFit parameter tensors in image_encoder.")
            
            # Enable only BitFit (bias) parameters
            for param in bitfit_params:
                param.requires_grad_(True)
                
            # Explicitly freeze non-bias parameters in image_encoder
            for name, param in self.model.image_encoder.named_parameters():
                if not any('bias' in name and param is bitfit_param for bitfit_param in bitfit_params):
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            for name in name_to_update:
                if hasattr(self.model, name):
                    load_pretrained_weights(self.model.__getattr__(name), cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Build optimizers
        self.optimizers = {}
        for name in name_to_update:
            if hasattr(self.model, name):
                module = self.model.__getattr__(name)
                
                # Special case: for BitFit image_encoder, optimize only BitFit parameters
                if name == "image_encoder" and has_bitfit_image_encoder:
                    bitfit_params = module.get_bitfit_parameters()
                    self.optimizers[name] = build_optimizer(bitfit_params, cfg.OPTIM)
                    print(f"Created optimizer for image_encoder with {len(bitfit_params)} BitFit parameter tensors")
                # Special case: for BitFit text_encoder, optimize only BitFit parameters
                elif name == "text_encoder" and has_bitfit_text_encoder:
                    bitfit_params = module.get_bitfit_parameters()
                    self.optimizers[name] = build_optimizer(bitfit_params, cfg.OPTIM)
                    print(f"Created optimizer for text_encoder with {len(bitfit_params)} BitFit parameter tensors")
                else:
                    # Regular modules: optimize all their parameters
                    trainable_params = [p for p in module.parameters() if p.requires_grad]
                    if trainable_params:  # Only create optimizer if there are trainable params
                        self.optimizers[name] = build_optimizer(trainable_params, cfg.OPTIM)
        
        self.schedulers = {name: build_lr_scheduler(optim, cfg.OPTIM) for name, optim in self.optimizers.items()}

        for name in name_to_update:
            if hasattr(self.model, name) and name in self.optimizers:
                self.register_model(name, self.model.__getattr__(name), self.optimizers[name], self.schedulers[name])

        print(f"Registered models: {[name for name in self._models]}")
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.4f} of total)")
        
        # Print BitFit-specific statistics if available
        if has_bitfit_image_encoder:
            bitfit_params = self.model.image_encoder.get_bitfit_parameters()
            bitfit_param_count = sum(p.numel() for p in bitfit_params)
            print(f"BitFit parameters in image_encoder: {bitfit_param_count:,} ({bitfit_param_count/total_params:.4f} of total)")
            frozen_encoder_params = sum(p.numel() for name, p in self.model.image_encoder.named_parameters() 
                                       if not any(p is bitfit_param for bitfit_param in bitfit_params))
            print(f"Frozen parameters in image_encoder: {frozen_encoder_params:,}")
            
        if has_bitfit_text_encoder:
            bitfit_params = self.model.text_encoder.get_bitfit_parameters()
            bitfit_param_count = sum(p.numel() for p in bitfit_params)
            print(f"BitFit parameters in text_encoder: {bitfit_param_count:,} ({bitfit_param_count/total_params:.4f} of total)")
            frozen_encoder_params = sum(p.numel() for name, p in self.model.text_encoder.named_parameters() 
                                       if not any(p is bitfit_param for bitfit_param in bitfit_params))
            print(f"Frozen parameters in text_encoder: {frozen_encoder_params:,}")
            

        self.scaler = GradScaler() if cfg.TRAINER.MODEL.PREC == "amp" else None

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.05)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.wassertein_1_loss = Wasserstein1Loss()

        self.centers = None
        self.uncertainty_centers = None

    def forward_backward(self, batch_x, batch_u):
        entropy = EntropyLoss()
        point_cloud, label = self.parse_batch_train(batch_x)
        point_cloud_unlabelled, label_unlabelled = self.parse_batch_train(batch_u)

        model = self.model
        optimizers = self.optimizers
        scaler = self.scaler


        prec = self.cfg.TRAINER.MODEL.PREC
        
        source_logits, loss_cls, loss_transform, loss_align, entropy_loss_labelled = model(point_cloud, label)
        target_logits, loss_transform_2, loss_align_2, entropy_loss_unlabelled = model(point_cloud_unlabelled)
        # Compute target uncertainty-aware losses
        target_features = model.get_image_features(point_cloud_unlabelled)
        prototype_loss, similarity_weights, pseudo_labels = self.compute_target_uncertainty_loss(target_features)
        loss_cls_target = torch.tensor(0) if pseudo_labels is None else F.cross_entropy(target_logits, pseudo_labels)

        loss_entropy =  entropy(target_logits.softmax(dim=-1))
        loss_sinkhorn = self.sinkhorn_loss(target_logits.softmax(dim = -1), source_logits.softmax(dim = -1).detach())
        loss_wasserstein = self.wassertein_1_loss(target_logits.softmax(dim=-1), source_logits.softmax(dim=-1).detach())
        loss_kl = self.kl_loss(target_logits.log_softmax(dim = -1), source_logits.softmax(dim = -1).detach())  # taking log_softmax for source and normal for target

        loss = loss_cls + loss_transform + loss_transform_2 + entropy_loss_labelled + entropy_loss_unlabelled
        
        if self.cfg.TRAINER.USE_PROTOTYPE_LOSS:
            loss += prototype_loss
        if self.cfg.TRAINER.USE_SINKHORN_LOSS:
            loss += loss_sinkhorn
        if self.cfg.TRAINER.USE_ENTROPY_LOSS:
            loss += loss_entropy
        if self.cfg.TRAINER.USE_ALIGN_LOSS:
            loss += loss_align
            loss += loss_align_2
        if self.cfg.TRAINER.USE_KL_LOSS:
            loss += loss_kl
        if self.cfg.TRAINER.USE_W1_LOSS:
            loss += loss_wasserstein


        for optim in optimizers.values():
            optim.zero_grad()
        
        loss.backward()
        for optim in optimizers.values():
            optim.step()

        loss_summary = {
            "loss": loss.item(), 
            "loss_cls": loss_cls.item(), 
            "loss_cls_target": loss_cls_target.item(),
            "loss_entropy":loss_entropy.item(), 
            'loss_transformation': (loss_transform + loss_transform_2).item(), 
            'loss_sinkhorn': loss_sinkhorn.item(), 
            'loss_kl':loss_kl.item(),
            'loss_prototype': prototype_loss.item(),
            'loss_wasserstein':loss_wasserstein.item(),
            'loss_align': (loss_align + loss_align_2).item(),
            'entropy_loss_labelled': entropy_loss_labelled.item(),
            'entropy_loss_unlabelled': entropy_loss_unlabelled.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch):
        point_cloud, label = batch
        point_cloud = point_cloud.to(self.device)
        label = label.to(self.device)

        return point_cloud, label

    def parse_batch_test(self, batch):
        point_cloud,  label = batch
        point_cloud = point_cloud.to(self.device)
        label = label.to(self.device)

        return point_cloud, label
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            point_cloud, label = self.parse_batch_test(batch)
            output = self.model(point_cloud=point_cloud)[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def build_data_loader(self):
        if self.cfg.DATASET.NAME == 'PointDA':
            source_domain = self.cfg.DATASET.SOURCE_DOMAINS[0]
            target_domain = self.cfg.DATASET.TARGET_DOMAINS[0]
            if source_domain == 'scannet':
                train_dataset = Scannet(f'{self.cfg.DATASET.ROOT}/{source_domain}', split='train')
            else:
                train_dataset = PointDA(f'{self.cfg.DATASET.ROOT}/{source_domain}', split='train')

            
            if target_domain == 'scannet':
                train_dataset_unlabelled = Scannet(f'{self.cfg.DATASET.ROOT}/{target_domain}', split='train')
                test_dataset = Scannet(f'{self.cfg.DATASET.ROOT}/{target_domain}', split='test')
            else:
                train_dataset_unlabelled = PointDA(f'{self.cfg.DATASET.ROOT}/{target_domain}', split='train')
                test_dataset = PointDA(f'{self.cfg.DATASET.ROOT}/{target_domain}', split='test')

            

            print(f"Length of train dataset: {len(train_dataset)}")
            print(f"Length of unlabelled train dataset: {len(train_dataset_unlabelled)}")
            print(f"Length of test dataset: {len(test_dataset)}")

            self.train_loader_x = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.train_loader_u = torch.utils.data.DataLoader(
                train_dataset_unlabelled,
                batch_size=self.cfg.DATALOADER.TRAIN_U.BATCH_SIZE, 
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.test_loader  = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.num_classes = len(train_dataset.idx_to_class)
            self.num_source_domains = 1
            self.lab2cname = train_dataset.idx_to_class

        elif self.cfg.DATASET.NAME == 'GraspNet':
            source_domain = self.cfg.DATASET.SOURCE_DOMAINS[0]
            target_domain = self.cfg.DATASET.TARGET_DOMAINS[0]
            if source_domain == 'Synthetic':
                train_dataset = GraspNet(self.cfg.DATASET.ROOT, split='train', data_type='Synthetic')
            elif source_domain == 'kinect':
                train_dataset = GraspNet(self.cfg.DATASET.ROOT, split='train', data_type='Real', camera='kinect')
            elif source_domain == 'realsense':
                train_dataset = GraspNet(self.cfg.DATASET.ROOT, split='train', data_type='Real', camera='realsense')

            if target_domain == 'Synthetic':
                raise ValueError("Synthetic target domain not supported")
            elif target_domain == 'kinect':
                train_dataset_unlabelled = GraspNet(self.cfg.DATASET.ROOT, split='train', data_type='Real', camera='kinect')
                test_dataset = GraspNet(self.cfg.DATASET.ROOT, split='test', data_type='Real', camera='kinect')
            elif target_domain == 'realsense':
                train_dataset_unlabelled = GraspNet(self.cfg.DATASET.ROOT, split='train', data_type='Real', camera='realsense')
                test_dataset = GraspNet(self.cfg.DATASET.ROOT, split='test', data_type='Real', camera='realsense')

            print(f"Length of train dataset: {len(train_dataset)}")
            print(f"Length of unlabelled train dataset: {len(train_dataset_unlabelled)}")
            print(f"Length of test dataset: {len(test_dataset)}")

            self.train_loader_x = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.train_loader_u = torch.utils.data.DataLoader(
                train_dataset_unlabelled,
                batch_size=self.cfg.DATALOADER.TRAIN_U.BATCH_SIZE, 
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.test_loader  = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA)
            )

            self.num_classes = len(train_dataset.idx_to_class)
            self.num_source_domains = 1
            self.lab2cname = train_dataset.idx_to_class

    def compute_confidence_score(self, logits, num_classes):
        """
        Compute confidence score based on entropy: w_i^s = 1 - H(softmax(f_θ(x_i^s))) / log(C)
        
        Args:
            logits (torch.Tensor): Model logits/predictions
            num_classes (int): Total number of classes C
            
        Returns:
            torch.Tensor: Confidence scores
        """
        # Compute softmax probabilities from logits
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy: H(p) = -∑ p_c * log(p_c)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Normalize by log(C) and compute confidence: w = 1 - H/log(C)
        max_entropy = math.log(num_classes)
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence

    def after_epoch_uncertainty(self, num_subsets=1):
        """
        Compute uncertainty-aware class prototypes using weighted sub-prototypes.
        
        Args:
            num_subsets (int): Number of subsets to divide each class into
        """
        self.model.eval()
        
        # Collect all features and predictions per class
        class_data = {}  # class_idx -> [(feature, prediction), ...]
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader_x):
                point_cloud, labels = self.parse_batch_train(batch)
                point_cloud = point_cloud.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions (logits) and image features
                predictions = self.model(point_cloud)[0]
                image_features = self.model.get_image_features(point_cloud)
                
                # Store data per class
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    if label_idx not in class_data:
                        class_data[label_idx] = []
                    
                    class_data[label_idx].append((
                        image_features[i].cpu(),
                        predictions[i].cpu()
                    ))
        
        # Determine dimensions
        max_class_idx = max(class_data.keys()) if class_data else 0
        num_classes = max_class_idx + 1
        feature_dim = class_data[list(class_data.keys())[0]][0][0].shape[0] if class_data else 512
        
        # Initialize uncertainty-aware prototypes array
        uncertainty_prototypes = torch.zeros(num_classes, feature_dim)
        
        for class_idx in class_data:
            if len(class_data[class_idx]) == 0:
                continue
                
            # Extract features and predictions for this class
            features_list = [item[0] for item in class_data[class_idx]]
            predictions_list = [item[1] for item in class_data[class_idx]]
            
            features = torch.stack(features_list)  # Shape: [N, feature_dim]
            predictions = torch.stack(predictions_list)  # Shape: [N, num_classes]

            # Compute confidence scores
            confidence_scores = self.compute_confidence_score(predictions, self.num_classes)
            
            # Divide into random subsets
            num_samples = len(features)
            indices = torch.randperm(num_samples)
            
            # Calculate samples per subset
            num_subsets = num_samples
            samples_per_subset = num_samples // num_subsets
            if samples_per_subset == 0:
                # If too few samples, use all samples as one subset
                subset_indices = [indices]
                actual_num_subsets = 1
            else:
                subset_indices = []
                for j in range(num_subsets):
                    start_idx = j * samples_per_subset
                    if j == num_subsets - 1:
                        # Last subset gets remaining samples
                        end_idx = num_samples
                    else:
                        end_idx = (j + 1) * samples_per_subset
                    subset_indices.append(indices[start_idx:end_idx])
                actual_num_subsets = num_subsets
            
            # Compute weighted sub-prototypes
            sub_prototypes = []
            for subset_idx in subset_indices:
                if len(subset_idx) == 0:
                    continue
                    
                # Get features and weights for this subset
                subset_features = features[subset_idx]  # Shape: [subset_size, feature_dim]
                subset_weights = confidence_scores[subset_idx]  # Shape: [subset_size]
                
                # Compute weighted prototype: p_c^(j) = Σ(w_i^s * f_θ(x_i^s)) / Σ(w_i^s)
                weighted_sum = (subset_features.T * subset_weights).T.sum(dim=0)  # Shape: [feature_dim]
                weight_sum = subset_weights.sum()
                
                if weight_sum > 0:
                    sub_prototype = weighted_sum / weight_sum
                else:
                    # Fallback to simple average if all weights are zero
                    sub_prototype = subset_features.mean(dim=0)
                
                sub_prototypes.append(sub_prototype)
            
            # Compute final prototype as average of sub-prototypes
            if len(sub_prototypes) > 0:
                uncertainty_prototypes[class_idx] = torch.stack(sub_prototypes).mean(dim=0)
                # print(f"Class {class_idx}: {num_samples} samples, {actual_num_subsets} subsets, "
                #       f"avg confidence: {confidence_scores.mean():.3f}")
        
        # print(f"Uncertainty prototypes shape: {uncertainty_prototypes.shape}")
        self.uncertainty_centers = uncertainty_prototypes

    def compute_target_uncertainty_loss(self, target_features):
        """
        Compute uncertainty-aware alignment loss for target domain samples.
        Uses pseudo-labels generated from similarity to prototypes.
        
        Args:
            target_features (torch.Tensor): Target domain image features [B, feature_dim]
            
        Returns:
            tuple: (prototype_loss, similarity_weights)
                - prototype_loss: Uncertainty-aware prototype alignment loss  
                - similarity_weights: Alpha weights for other losses [B]
        """
        # if self.uncertainty_centers is None:
        #     # Fallback to regular centers if uncertainty centers not available
        #     prototypes = self.centers if self.centers is not None else None
        # else:
        prototypes = self.uncertainty_centers if self.uncertainty_centers is not None else None
            
        if prototypes is None:
            # Return zero loss if no prototypes available
            return torch.tensor(0.0, device=target_features.device), torch.ones(target_features.size(0), device=target_features.device), None
        
        prototypes = prototypes.to(target_features.device)
        B, feature_dim = target_features.shape
        num_classes = prototypes.shape[0]
        
        # Normalize features and prototypes for cosine similarity
        target_features_norm = F.normalize(target_features, p=2, dim=-1)  # [B, feature_dim]
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)  # [num_classes, feature_dim]
        
        # Compute cosine similarities: s_i = cos(f_θ(x_i^t), p_c) for all classes
        cos_similarities = target_features_norm @ prototypes_norm.T  # [B, num_classes]
        
        # Generate pseudo-labels from similarities (most similar prototype)
        pseudo_labels = cos_similarities.argmax(dim=1)  # [B]
        
        # 3) Similarity weight estimation: α_i = (s_i + 1) / 2 ∈ [0, 1]
        # From the pseudo-labeled class c
        pseudo_similarities = cos_similarities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)  # [B]
        similarity_weights = (pseudo_similarities + 1) / 2  # [B] ∈ [0, 1]
        
        # 4) Compute target uncertainty weights: w_i^t = 1 - H(softmax(cos(f_θ(x_i^t), p_c))) / log(C)  
        target_confidence_scores = self.compute_confidence_score(cos_similarities, num_classes)  # [B]
        
        # Compute uncertainty-aware prototype alignment loss:
        # L_proto = -∑_i w_i^t * log(softmax(cos(f_θ(x_i^t), p_c)))
        log_probs = F.log_softmax(cos_similarities, dim=-1)  # [B, num_classes]
        target_log_probs = log_probs.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)  # [B]
        
        # Weighted prototype loss
        prototype_loss = -(target_confidence_scores * target_log_probs).mean()

        
        return prototype_loss, similarity_weights, pseudo_labels

    def after_epoch_centers(self):
        centers = {}
        class_features = {}
        class_counts = {}
        
        # print("Computing class centers...")
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader_x):                
                # if batch_idx % 10 == 0:
                #     print(f"Processing batch {batch_idx}/{len(self.train_loader_x)}")
                
                # Unpack batch data
                point_cloud, labels = self.parse_batch_train(batch)
                
                # Move to device
                point_cloud = point_cloud.to(self.device)
                labels = labels.to(self.device)
                
                # Get image features using the model's method
                image_features = self.model.get_image_features(point_cloud)
                
                # Accumulate features by class
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    
                    if label_idx not in class_features:
                        class_features[label_idx] = []
                        class_counts[label_idx] = 0
                    
                    class_features[label_idx].append(image_features[i].cpu())
                    class_counts[label_idx] += 1
        
        # Determine max class index to create array
        max_class_idx = max(class_features.keys()) if class_features else 0
        num_classes = max_class_idx + 1
        
        # Initialize centers array with zeros
        feature_dim = class_features[list(class_features.keys())[0]][0].shape[0] if class_features else 512
        centers_array = torch.zeros(num_classes, feature_dim)
        
        # Compute centers as average of features for each class
        for class_idx in class_features:
            if len(class_features[class_idx]) > 0:
                centers_array[class_idx] = torch.stack(class_features[class_idx]).mean(dim=0)
                # print(f"Class {class_idx}: {class_counts[class_idx]} samples")
        
        # print(f"Centers array shape: {centers_array.shape}")
        self.centers = centers_array
        
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        self.after_epoch_centers()
        self.after_epoch_uncertainty()
