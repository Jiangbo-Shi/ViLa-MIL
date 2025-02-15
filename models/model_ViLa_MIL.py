# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


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
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1:-n_ctx, :]) 

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  
                    ctx,     
                    suffix, 
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
                        prefix_i,     
                        ctx_i_half1,  
                        class_i,      
                        ctx_i_half2,  
                        suffix_i,   
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
                        prefix_i,  
                        class_i,   
                        ctx_i,    # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        clip_model, _ = clip.load("RN50", device="cpu")
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float())

        self.norm = nn.LayerNorm(config.input_size)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)

        self.learnable_image_center = nn.Parameter(torch.Tensor(*[config.prototype_number, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=.02)

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M) 
        compents = self.norm(compents + self.learnable_image_center)

        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)

        H = compents.squeeze().float()
        A_V = self.attention_V(H)  
        A_U = self.attention_U(H)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  
        A = F.softmax(A, dim=1)  
        image_features_low = torch.mm(A, H)  

        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)  
        A_U_high = self.attention_U(H_high)  
        A_high = self.attention_weights(A_V_high * A_U_high) 
        A_high = torch.transpose(A_high, 1, 0)  
        A_high = F.softmax(A_high, dim=1)  
        image_features_high = torch.mm(A_high, H_high)  

        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M), dim=0)
        text_context_features, _ = self.cross_attention_2(text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_features.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_features_high.squeeze() + text_features_high

        logits_low = image_features_low @ text_features_low.T.cuda()
        logits_high = image_features_high @ text_features_high.T.cuda()
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]

        return Y_prob, Y_hat, loss

