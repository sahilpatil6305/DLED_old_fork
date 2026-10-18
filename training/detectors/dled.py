import os
import datetime
import logging
import copy
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from loralib.utils import apply_lora, apply_double_lora, INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION

_tokenizer = _Tokenizer()
logger = logging.getLogger(__name__)

CUSTOM_TEMPLATES = {    
    'deepfake': 'a {} photo.',
    'deepfake1': 'a {} human face photo.',
    'multi': 'a {} photo.',
}

def load_clip_to_cpu(config):
    backbone_name = config['MODEL']['BACKBONE']['NAME']
    url = clip._MODELS[backbone_name]
    # Directory used to cache the CLIP weights. Override with the
    # DLED_PRETRAINED_ROOT environment variable.
    root_directory = os.environ.get(
        'DLED_PRETRAINED_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'pretrained', 'clip'),
    )
    os.makedirs(root_directory, exist_ok=True)
    model_path = clip._download(url, root=root_directory)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())

    return model

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def _validate_lora_checkpoint_compatibility(model_state, checkpoint_state):
    """Reject checkpoints whose LoRA target keys or adapter shapes differ."""
    expected = {key: value for key, value in model_state.items() if 'lora_' in key}
    supplied = {key: value for key, value in checkpoint_state.items() if 'lora_' in key}
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    shape_mismatches = sorted(
        (key, tuple(supplied[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(supplied)
        if supplied[key].shape != expected[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "LoRA checkpoint is incompatible with this configured adapter variant. "
            f"Missing keys ({len(missing)}): {missing[:12]}; "
            f"unexpected keys ({len(unexpected)}): {unexpected[:12]}; "
            f"shape mismatches (key, checkpoint, model) ({len(shape_mismatches)}): "
            f"{shape_mismatches[:12]}. Checkpoints from a different LoRA rank/target "
            "variant must not be loaded."
        )

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['TRAINER']['N_CTX']
        ctx_init = cfg['TRAINER']['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg['resolution']
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if cfg['TRAINER']['CSC']:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad(): 
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # (n_cls, 77, dim)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS    (n_cls, 1 ,dim)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS    (n_cls, 60, dim)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg['TRAINER']['CLASS_TOKEN_POSITION']


    def forward(self):
        ctx = self.ctx     # (n_ctx, ctx_dim)
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, spatial_clip_model, frequency_clip_model, device, act_out='softplus'):
        super().__init__()
        evidence_map={
            'exp':'exp',
            'relu':'relu',
            'softplus':'softplus'
        }
        self.num_classes = len(classnames)

        self.spatial_image_encoder = spatial_clip_model.visual
        self.spatial_text_encoder = TextEncoder(spatial_clip_model)
        self.spatial_prompt_learner = PromptLearner(cfg, classnames, spatial_clip_model, device)
        self.spatial_tokenized_prompts = self.spatial_prompt_learner.tokenized_prompts
        self.spatial_logit_scale = spatial_clip_model.logit_scale

        self.frequency_image_encoder = frequency_clip_model.visual
        self.frequency_text_encoder = TextEncoder(frequency_clip_model)
        self.frequency_prompt_learner = PromptLearner(cfg, classnames, frequency_clip_model, device)
        self.frequency_tokenized_prompts = self.frequency_prompt_learner.tokenized_prompts
        self.frequency_logit_scale = frequency_clip_model.logit_scale
        
        self.dtype = spatial_clip_model.dtype
        self.evidence_func = eval(evidence_map[act_out] + '_evidence')
        print("evidence function: ", act_out)
        self.act_out = act_out
        self.device = device
        self.lora_layers = []
        text_indices = INDEX_POSITIONS_TEXT[cfg['TRAINER']['POSITION']]
        visual_indices = INDEX_POSITIONS_VISION[cfg['MODEL']['BACKBONE']['NAME']][cfg['TRAINER']['POSITION']]

        self.spatial_text_lora_layers = apply_lora(cfg, self.spatial_text_encoder, text_indices, self.dtype)
        self.lora_layers.extend(self.spatial_text_lora_layers)

        self.spatial_visual_lora_layers = apply_lora(cfg, self.spatial_image_encoder, visual_indices, self.dtype)
        self.lora_layers.extend(self.spatial_visual_lora_layers)
    
        self.frequency_text_lora_layers = apply_lora(cfg, self.frequency_text_encoder, text_indices, self.dtype)
        self.lora_layers.extend(self.frequency_text_lora_layers)

        self.frequency_visual_lora_layers = apply_lora(cfg, self.frequency_image_encoder, visual_indices, self.dtype)
        self.lora_layers.extend(self.frequency_visual_lora_layers)

    def forward(self, data_dict):
        freq_map = data_dict['freq']
        image = data_dict['image']
        n_cls = self.num_classes

        spatial_prompts= self.spatial_prompt_learner()
        spatial_tokenized_prompts = self.spatial_tokenized_prompts
        frequency_prompts= self.frequency_prompt_learner()
        frequency_tokenized_prompts = self.frequency_tokenized_prompts

        visual_features = []
        context_features = []

        """
        Get evidence from spatial domain
        """
        spatial_image_features = self.spatial_image_encoder(image.type(self.dtype))
        spatial_text_features = self.spatial_text_encoder(spatial_prompts, spatial_tokenized_prompts)

        spatial_image_features = spatial_image_features / spatial_image_features.norm(dim=-1, keepdim=True)
        spatial_text_features = spatial_text_features / spatial_text_features.norm(dim=-1, keepdim=True)

        spatial_logit_scale = self.spatial_logit_scale.exp()
        spatial_logits = spatial_logit_scale * spatial_image_features @ spatial_text_features.t()

        # produce evidence and uncertainty
        spatial_evidence = self.evidence_func(spatial_logits)
        spatial_alpha = spatial_evidence + 1.0
        spatial_total_strength = torch.sum(spatial_alpha, dim=1, keepdim=True)  # (B, 1)
        spatial_pred = spatial_alpha / spatial_total_strength
        spatial_vacuity_uncertainty = self.num_classes / spatial_total_strength.squeeze(-1)

        output_dict = {'spatial_pred_mean': spatial_pred,
                       'spatial_pred_alpha': spatial_alpha,
                       'spatial_evidence': spatial_evidence,
                       'spatial_vacuity': spatial_vacuity_uncertainty}
        visual_features.append(spatial_image_features)
        context_features.append(spatial_text_features)

        """
        Get evidence from frequency domain
        """
        frequency_image_features = self.frequency_image_encoder(freq_map.type(self.dtype))
        frequency_text_features = self.frequency_text_encoder(frequency_prompts, frequency_tokenized_prompts)

        frequency_image_features = frequency_image_features / frequency_image_features.norm(dim=-1, keepdim=True)
        frequency_text_features = frequency_text_features / frequency_text_features.norm(dim=-1, keepdim=True)

        frequency_logit_scale = self.frequency_logit_scale.exp()
        frequency_logits = frequency_logit_scale * frequency_image_features @ frequency_text_features.t()

        # produce evidence and uncertainty
        frequency_evidence = self.evidence_func(frequency_logits)
        frequency_alpha = frequency_evidence + 1.0
        frequency_total_strength = torch.sum(frequency_alpha, dim=1, keepdim=True)  # (B, 1)
        frequency_pred = frequency_alpha / frequency_total_strength
        frequency_vacuity_uncertainty = self.num_classes / frequency_total_strength.squeeze(-1)

        output_dict.update({'frequency_pred_mean': frequency_pred,
                            'frequency_pred_alpha': frequency_alpha,
                            'frequency_evidence': frequency_evidence,
                            'frequency_vacuity': frequency_vacuity_uncertainty})
        visual_features.append(frequency_image_features)
        context_features.append(frequency_text_features)

        """
        Fuse evidence 
        """
        # Get conflict vector
        spatial_belief = spatial_evidence / spatial_total_strength
        frequency_belief = frequency_evidence / frequency_total_strength
        spatial_belief_expanded = spatial_belief.unsqueeze(2)  # 32*4*1
        frequency_belief_expanded = frequency_belief.unsqueeze(1)  # 32*1*4
        
        product = spatial_belief_expanded * frequency_belief_expanded  # 32*4*4
        mask = (torch.ones(n_cls, n_cls) - torch.eye(n_cls)).to(self.device)  # create a mask to mask j==k, 1*4*4 / 1*2*2
        masked_product = product * mask  # 32*4*4/ 32*2*2
        conflict_vec = masked_product.sum(dim=(1, 2)).unsqueeze(1)  # 32*1

        #Get updated belief
        fused_belief = (spatial_belief * frequency_belief + spatial_belief * frequency_vacuity_uncertainty.unsqueeze(1) + frequency_belief * spatial_vacuity_uncertainty.unsqueeze(1)) / (1-conflict_vec)
        fused_vacuity_uncertainty = ((spatial_vacuity_uncertainty.unsqueeze(1) * frequency_vacuity_uncertainty.unsqueeze(1)) /(1 - conflict_vec))
        fused_total_strength = self.num_classes / fused_vacuity_uncertainty
        fused_evidence = fused_belief * fused_total_strength
        fused_alpha = fused_evidence + 1
        fused_pred = fused_alpha / fused_total_strength
        max_fused_evi_val, _ = torch.max(fused_evidence, dim=1)
        fused_vacuity_uncertainty = 1 / (1+max_fused_evi_val)

        output_dict.update({'pred_mean': fused_pred,
                            'pred_alpha': fused_alpha,
                            'evidence': fused_evidence,
                            'vacuity': fused_vacuity_uncertainty,
                            'image': image,
                            'freq': freq_map
                            })
        return output_dict, visual_features, context_features


    def compute_dissonance(self, evidence):
        """ evidence: (B, K)
        """
        def Bal(b_i, b_j):
            return 1 - torch.abs(b_i - b_j) / (b_i + b_j)
        
        total_strength = torch.sum(evidence + 1.0, dim=1, keepdim=True)
        belief = evidence / total_strength
        dissonance = 0.0
        for i in range(self.num_classes):
            bi = belief[:, i]
            term_Bal, term_bj = 0.0, 0.0
            for j in range(self.num_classes):
                if j != i:
                    bj = belief[:, j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dissonance += bi * term_Bal / term_bj
        return dissonance


@DETECTOR.register_module(module_name='dled')
class DLED_Detector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.training_mode = config['training_mode']
        print("Train with: ", self.training_mode, " Forgery Category!")
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        if torch.cuda.is_available():
            if self.cfg['device_choice'] == 'new':
                local_rank = int(os.environ.get(
                    "LOCAL_RANK", self.cfg.get('local_rank', 0)
                ))
                self.device = torch.device("cuda", local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def check_cfg(self, cfg):
        assert cfg['TRAINER']['PREC'] in ["fp16", "fp32", "amp"]

    def build_backbone(self, cfg):
        if self.training_mode == 'single':
            classnames = ['real', 'fake']
        elif self.training_mode == 'multi':
            training_set_name = cfg['train_dataset']
            if cfg['backbone_config']['num_classes'] == 2:
                if training_set_name[0] in ["DF40_FF_train_EFS", "DF40_mixed_train_EFS"]:
                    classnames = ['real', 'synthesis-fake']
                elif training_set_name[0] in ["DF40_FF_train_FS", "DF40_mixed_train_FS"]:
                    classnames = ['real', 'swapping-fake']
                elif training_set_name[0] in ["DF40_FF_train_FR", "DF40_mixed_train_FR"]:
                    classnames = ['real', 'reenactment-fake']
            elif cfg['backbone_config']['num_classes'] == 3:
                if training_set_name[0] in ["DF40_FF_train_NOEFS", "DF40_mixed_train_NOEFS"]:
                    classnames = ['real', 'swapping-fake', 'reenactment-fake']
                elif training_set_name[0] in ["DF40_FF_train_NOFS", "DF40_mixed_train_NOFS"]:
                    classnames = ['real', 'synthesis-fake', 'reenactment-fake']
                elif training_set_name[0] in ["DF40_FF_train_NOFR", "DF40_mixed_train_NOFR"]:
                    classnames = ['real', 'swapping-fake', 'synthesis-fake']
            else:
                classnames = ['real', 'swapping-fake', 'reenactment-fake', 'synthesis-fake']

        print("classnames: ", classnames)
        print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['NAME']})")
        spatial_clip_model = load_clip_to_cpu(cfg)
        frequency_clip_model = copy.deepcopy(spatial_clip_model)
        
        if cfg['TRAINER']['PREC'] == "fp32" or cfg['TRAINER']['PREC'] == "amp":
            # CLIP's default precision is fp16
            spatial_clip_model.float()
            frequency_clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, spatial_clip_model, frequency_clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["spatial_prompt_learner", "frequency_prompt_learner", "lora_"]
            
        for name, param in self.model.named_parameters():
            set_zero_grad = 0
            for up_name in name_to_update:
                if up_name in name:
                    set_zero_grad = 1
                    break
            if set_zero_grad == 0:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg['MODEL']['INIT_WEIGHTS']:
            load_pretrained_weights(self.model.spatial_prompt_learner, cfg['MODEL']['INIT_WEIGHTS'])
            load_pretrained_weights(self.model.frequency_prompt_learner, cfg['MODEL']['INIT_WEIGHTS'])

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.spatial_prompt_learner, cfg['OPTIM'])
        self.sched = build_lr_scheduler(self.optim, cfg['OPTIM'])
        self.register_model("spatial_prompt_learner", self.model.spatial_prompt_learner, self.optim, self.sched)
    
        self.spatial_text_lora_optim = build_optimizer(self.model.spatial_text_encoder, cfg['OPTIM'])
        self.spatial_text_lora_sched = build_lr_scheduler(self.spatial_text_lora_optim, cfg['OPTIM'])
        self.register_model("spatial_text_lora", self.model.spatial_text_encoder, self.spatial_text_lora_optim, self.spatial_text_lora_sched)

        self.spatial_visual_lora_optim = build_optimizer(self.model.spatial_image_encoder, cfg['OPTIM'])
        self.spatial_visual_lora_sched = build_lr_scheduler(self.spatial_visual_lora_optim, cfg['OPTIM'])
        self.register_model("spatial_visual_lora", self.model.spatial_image_encoder, self.spatial_visual_lora_optim, self.spatial_visual_lora_sched)

        self.freq_optim = build_optimizer(self.model.frequency_prompt_learner, cfg['OPTIM'])
        self.freq_sched = build_lr_scheduler(self.freq_optim, cfg['OPTIM'])
        self.register_model("frequency_prompt_learner", self.model.frequency_prompt_learner, self.freq_optim, self.freq_sched)
    
        self.frequency_text_lora_optim = build_optimizer(self.model.frequency_text_encoder, cfg['OPTIM'])
        self.frequency_text_lora_sched = build_lr_scheduler(self.frequency_text_lora_optim, cfg['OPTIM'])
        self.register_model("frequency_text_lora", self.model.frequency_text_encoder, self.frequency_text_lora_optim, self.frequency_text_lora_sched)

        self.frequency_visual_lora_optim = build_optimizer(self.model.frequency_image_encoder, cfg['OPTIM'])
        self.frequency_visual_lora_sched = build_lr_scheduler(self.frequency_visual_lora_optim, cfg['OPTIM'])
        self.register_model("frequency_visual_lora", self.model.frequency_image_encoder, self.frequency_visual_lora_optim, self.frequency_visual_lora_sched)
        self.scaler = GradScaler() if cfg['TRAINER']['PREC'] == "amp" else None

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class(config, config['num_classes'])
        return loss_func

    def features(self, data_dict: dict):
        return 

    def classifier(self, features: torch.tensor):
        return

    def forward(self, data_dict: dict, domain) -> dict:
        prec = self.cfg['TRAINER']['PREC']
        if prec == "amp":
            with autocast():
                out_dict, _, _ = self.model(data_dict)
        else:
            out_dict, _, _ = self.model(data_dict)
        return out_dict

    def model_full_fp(self, data_dict: dict) -> dict:
        prec = self.cfg['TRAINER']['PREC']
        if prec == "amp":
            with autocast():
                out_dict, image_features, text_features = self.model(data_dict)
        else:
            out_dict, image_features, text_features = self.model(data_dict)
        return out_dict
        
    def model_inference(self, data_dict, threshold):
        out_dict, image_features, text_features = self.model(data_dict)
        # get the probability of the pred
        spatial_pred_alpha = out_dict['spatial_pred_alpha']
        spatial_uncertainty = out_dict['spatial_vacuity']
        spatial_evidence = out_dict['spatial_evidence']
        _, spatial_prediction = torch.max(spatial_pred_alpha, 1)

        frequency_pred_alpha = out_dict['frequency_pred_alpha']
        frequency_uncertainty = out_dict['frequency_vacuity']
        frequency_evidence = out_dict['frequency_evidence']
        _, frequency_prediction = torch.max(frequency_pred_alpha, 1)

        fused_pred_alpha = out_dict['pred_alpha']
        fused_uncertainty = out_dict['vacuity']
        fused_evidence = out_dict['evidence']
        _, fused_prediction = torch.max(fused_pred_alpha, 1)

        prediction_dict = {'spatial':spatial_prediction, 'frequency':frequency_prediction, 'fused':fused_prediction}
        pred_alpha_dict = {'spatial':spatial_pred_alpha, 'frequency':frequency_pred_alpha, 'fused':fused_pred_alpha}
        uncertainty_dict = {'spatial':spatial_uncertainty, 'frequency':frequency_uncertainty, 'fused':fused_uncertainty}
        for domain_key in threshold.keys():
            # threshold.keys: spatial, frequency, fused
            sub_domain_uncertainty = threshold[domain_key]
            prediction = prediction_dict[domain_key]
            uncertainty = uncertainty_dict[domain_key]
            for uncer_key in sub_domain_uncertainty.keys():
                if uncer_key == 'overall':
                    continue
                uncertain_mask = prediction == uncer_key
                selected_uncertainty = uncertainty[uncertain_mask]
                uncertainty_threshold = threshold[domain_key][uncer_key]
                indices = torch.nonzero(selected_uncertainty > uncertainty_threshold)
                new_value = self.cfg['num_classes']
        
                p_count = 0
                for logit_index, p_logits in enumerate(prediction):
                    if uncertain_mask[logit_index]:
                        if p_count in indices:
                            prediction_dict[domain_key][logit_index] = new_value
                        p_count += 1
        pred_dict = {'prob': prediction_dict, 'pred_alpha': pred_alpha_dict, 'image_feat': image_features, 'uncertainty': uncertainty_dict}
        return pred_dict
    
    def get_losses(self, data_dict: dict, pred_dict: dict, mode='test') -> dict:
        label = data_dict['label']
        prec = self.cfg['TRAINER']['PREC']
        loss_dict = dict()
        if mode == 'train':
            cur_epoch = pred_dict['epoch']
            tot_epoch = pred_dict['total_epoch']
            if prec == "amp":
                with autocast():
                    spatial_loss_out = self.loss_func.separate_forward(pred_dict['spatial_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=True)
                    frequency_loss_out = self.loss_func.separate_forward(pred_dict['frequency_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=True)
            else:
                spatial_loss_out = self.loss_func.separate_forward(pred_dict['spatial_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=True)
                frequency_loss_out = self.loss_func.separate_forward(pred_dict['frequency_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=True)
            
        elif mode == 'test':
            # current epoch and total epoch are not used when testing
            cur_epoch, tot_epoch = -1, -1
            if prec == "amp":
                with autocast():
                    spatial_loss_out = self.loss_func.separate_forward(pred_dict['spatial_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=False)
                    frequency_loss_out = self.loss_func.separate_forward(pred_dict['frequency_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=False)
            else:
                spatial_loss_out = self.loss_func.separate_forward(pred_dict['spatial_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=False)
                frequency_loss_out = self.loss_func.separate_forward(pred_dict['frequency_pred_alpha'], label, cur_epoch, tot_epoch, isTraining=False)
            
        spatial_total_loss, frequency_total_loss = 0, 0
        for k, v in spatial_loss_out.items():
            if 'loss' in k:
                if 'avu' in k:
                    spatial_total_loss += v
                else:
                    spatial_total_loss += v
                loss_dict.update({"spatial_" + k: v})
        loss_dict.update({'spatial_overall': spatial_total_loss})

        for k, v in frequency_loss_out.items():
            if 'loss' in k:
                if 'avu' in k:
                    frequency_total_loss += v
                else:
                    frequency_total_loss += v
                loss_dict.update({"frequency_" + k: v})
        loss_dict.update({'frequency_overall': frequency_total_loss})
        loss_dict.update({'overall': spatial_total_loss + frequency_total_loss})
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['pred_alpha']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        return metric_batch_dict

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real
    
    def model_zero_grad(self, names=None):
        if self.cfg['TRAINER']['PREC'] == 'amp':
            self.optim.zero_grad()
        else:
            names = self.get_model_names(names)
            for name in names:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()

    def model_backward(self, loss):
        if self.cfg['TRAINER']['PREC'] == 'amp':
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def model_update(self, names=None):
        if self.cfg['TRAINER']['PREC'] == 'amp':
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            names = self.get_model_names(names)
            for name in names:
                if self._optims[name] is not None:
                    self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)
    
    def update_lr(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def load_model(self, model_path):
        if not model_path:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        ori_state_dict = torch.load(model_path, map_location=torch.device(self.device))
        if isinstance(ori_state_dict, dict) and isinstance(ori_state_dict.get('state_dict'), dict):
            ori_state_dict = ori_state_dict['state_dict']
        state_dict = {}
        for name, weights in ori_state_dict.items():
            new_key = name.replace("model.", "", 1) if name.startswith("model.") else name
            state_dict[new_key] = weights

        _validate_lora_checkpoint_compatibility(self.model.state_dict(), state_dict)
        print("keys: ", list(state_dict.keys())[0])
        self.model.load_state_dict(state_dict, strict=False)
