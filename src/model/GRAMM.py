import math
import torch
import torchaudio
import transformers
import random 

import pytorch_lightning as pl
import torch.nn as nn

from torch import Tensor
from typing import Optional, List, Tuple
from functools import partial
from .pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import DropPath
from timm.models.layers import trunc_normal_

from ..patching import PatchStrategy
from .utils import PatchEmbed, plot_fbank, repeat_token

from timm.models.vision_transformer import Block
from .layers import MWMHABlock

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import copy

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


from ..data_modules.scene_module import generate_scenes_batch
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

try:
    from einops import rearrange, repeat
except ImportError as e: 
    print(f"Got {e} this is expected if you are training.")


def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    print(block)
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


# It won't actually have the branching, because we already pad everything to 10 seconds.
pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)
collate_fn = torch.compile(collate_fn)
class GRAMM(pl.LightningModule):
    def __init__(self,
                lr=2e-4,
                trainer="adamW",
                b1=0.9,
                b2=0.95,
                weight_decay=0.0001,
                mask_patch=200,
                patch_strategy: PatchStrategy = None,
                in_channels : int = 2,
                num_mel_bins : int = 128,
                input_length : int = 1024,
                target_length : int = 200,
                sr : int = 32000,
                nr_samples_per_audio: int = 16,
                log_every_n_steps : int = 1000,
                encoder_embedding_dim: int = 768,
                encoder_depth=16,
                use_mwmae_decoder: bool = False,
                decoder_depth: int = 8,
                decoder_num_heads: int = 8,
                decoder_embedding_dim : int = 512,
                decoder_window_sizes: List[int] = [2, 5, 10, 25, 50, 100, 0, 0],
                mlp_ratio=4.,
                ssm_cfg=None,
                rms_norm: bool = False,
                residual_in_fp32: bool = False,
                fused_add_norm: bool = False,
                precision: str = "bf16-mixed",
                norm_epsilon: float = 1e-6,
                clean_data_ratio : float = 0.0,
                cluster: bool = False,
                **kwargs) -> None:

        super().__init__()
        self.clean_data_ratio = clean_data_ratio
        self.target_length = target_length
        self.input_length = input_length
        self.num_mel_bins = num_mel_bins 
        self.nr_samples_per_audio = nr_samples_per_audio
        self.sr = sr
        self.lr = lr 
        self.trainer_name = trainer 
        self.b1 = b1 
        self.b2 = b2 
        self.weight_decay = weight_decay
        self.mask_patch = mask_patch
        self.in_channels = in_channels
        self.log_every_n_steps = log_every_n_steps
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.patch_strategy = patch_strategy
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mlp_ratio = mlp_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_window_sizes = decoder_window_sizes
        self.encoder_cls_token_num = 1
        self.input_shape = None
        self.cluster = cluster 
        self.use_mwmae_decoder = use_mwmae_decoder

        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)

        
        # PatchEmbedding layer
        self.patch_embed = PatchEmbed()
        self._update_patch_embed_layers(self.patch_embed)
        
        # Pos Embed for the encoder.
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, self.num_patches + self.encoder_cls_token_num #For CLS Token
                , self.encoder_embedding_dim
            ),
            requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(self.encoder_embedding_dim, self.grid_size, cls_token_num=self.encoder_cls_token_num)
        self.encoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embedding_dim))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_norm = nn.LayerNorm(self.encoder_embedding_dim)

        self.encoder_v = MixerModel(
            d_model=encoder_embedding_dim,
            n_layer=encoder_depth,
            d_intermediate=0,
            ssm_cfg={"layer" : "Mamba2"},
            attn_layer_idx=None,
            attn_cfg=None,
            initializer_cfg=None,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
        )

        self.enc_drop_path = nn.Identity()

        self.decoder_embed = nn.Linear(self.encoder_embedding_dim, self.decoder_embedding_dim, bias=True) # Mapping from encoder to decoder
        # Mask token of the decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embedding_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        # Decoder last layer norm
        self.decoder_norm = nn.LayerNorm(self.decoder_embedding_dim)
        # Init the nn.Parameters here
        if self.use_mwmae_decoder:
            print("Using MWMAE Decoder")
            self.decoder_v =nn.ModuleList([MWMHABlock(
                dim=self.decoder_embedding_dim,
                num_heads=self.decoder_num_heads,
                window_sizes=self.decoder_window_sizes,
                shift_windows=False,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm) for i in range(self.decoder_depth)])
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_embedding_dim), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.decoder_embedding_dim, self.grid_size, cls_token_num=0)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        else:     
            self.decoder_v = nn.ModuleList([
            Block(self.decoder_embedding_dim, 
                    num_heads = self.decoder_num_heads, 
                    mlp_ratio = self.mlp_ratio, 
                    qkv_bias=True, 
                    norm_layer=nn.LayerNorm)
            for _ in range(self.decoder_depth)])
            # Pos Embed init with cls token num
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.encoder_cls_token_num, self.decoder_embedding_dim), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.decoder_embedding_dim, self.grid_size, cls_token_num=self.encoder_cls_token_num)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        if (self.in_channels == 2) or (self.in_channels == 1):
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=320,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float() 
        else:
            self.melspec = FeatureExtractor(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=self.sr // 100,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float()
        
        # Define prediction layers for Masked Auto Encoder pretraining
        self.spec_pred = nn.Sequential(
            nn.Linear(self.decoder_embedding_dim, 
                      self.patch_strategy.fshape * self.patch_strategy.tshape * self.in_channels,
                      bias = True),
        )

        # After applying the layer norm initialization, intialize the spectrogram.
        self.apply(
            partial(
                _init_weights,
                n_layer=self.encoder_depth,
            )
        )

        self.spectrogram_normalize = nn.LayerNorm(
                    [self.in_channels, self.num_mel_bins, self.target_length], 
                    elementwise_affine=False
                )
        self.input_shape = [self.num_mel_bins, self.target_length]

        compile_modules = kwargs.get("compile_modules", None)
        if (compile_modules is not None) and (compile_modules):
            self._compile_operations()

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.pass_through_decoder = torch.compile(self.pass_through_decoder, mode = "reduce-overhead")

        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False
        
    def _update_patch_embed_layers(self , patch_embed):
        """Updates the patch embedding and positional embedding layers."""
        # Update patch projection layer
        # Use 2, as the spectrogram has 2 channels
        patch_embed.proj = torch.nn.Conv2d(
            self.in_channels,
            self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
        )
        patch_embed.num_patch = self.num_patches


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.encoder_v)
        }
    

    def _forward_decoder_blocks(self, x, inference_params=None):
        hidden_states = x
        for layer in self.decoder_v:
            hidden_states = layer(hidden_states)
        hidden_states = self.decoder_norm(hidden_states)
        return hidden_states

    def pass_through_encoder(self, x, non_mask_index, B):
        """Passes the input through the Encoder Transformer network."""
        x = x + self.encoder_pos_embed[:, self.encoder_cls_token_num:, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = self.cls_token.expand(B, -1, -1) + self.encoder_pos_embed[:, :self.encoder_cls_token_num, :]
        x = torch.cat((cls_token, x), dim=1)
        return self.encoder_v(x, inference_params=None)

    def pass_through_encoder_until_block(self, x, non_mask_index, B, block_num = -1):
        """Passes the input through the Encoder Transformer network."""
        x = x + self.encoder_pos_embed[:, self.encoder_cls_token_num:, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = self.cls_token.expand(B, -1, -1) + self.encoder_pos_embed[:, :self.encoder_cls_token_num, :]
        x = torch.cat((cls_token, x), dim=1)
        return self._forward_encoder_blocks_until(x, block=block_num)
        
    def pass_through_decoder(self, encoder_output, non_mask_index, B):

        encoder_output = self.decoder_embed(encoder_output)
        x_ = repeat_token(
            self.mask_token, (B, self.num_patches)
        ).type_as(encoder_output)
        x_[non_mask_index, :] = encoder_output[
            :, self.encoder_cls_token_num :, :
        ].reshape((-1, encoder_output.shape[-1]))
        x_ = x_.reshape((B, -1, encoder_output.shape[-1]))
        
        # Concatenate the CLS and Possibly Distill tokens from the encoder
        if self.use_mwmae_decoder:
            x = x_
            return_cut = 0
        else:
            x = torch.cat([encoder_output[:, :self.encoder_cls_token_num, :], x_], dim=1)
            return_cut = self.encoder_cls_token_num
        x = x + self.decoder_pos_embed # add the pos embeds
        # Pass through transformer blocks
        x = self._forward_decoder_blocks(x)
        pred = self.spec_pred(x)
        pred = pred[:, return_cut:, :]
        return pred

    def _wav2fbank(self, waveform):
        with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation
            waveform = waveform.float()
            mel = self.melspec(waveform)  # Ensure input is float32
            if self.in_channels == 2:
                log_mel = torch.log(mel + 1e-5).transpose(3, 2)
            else:
                # Otherwise we have already the log mel spec.
                log_mel = mel.transpose(3,2)
        return log_mel

    @torch.no_grad()
    def _prepare_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Prepare the batch before the training step.
        This version is vectorized for performance and memory efficiency.
        '''
        audio, noise, source_rir, noise_rirs, snr, mask = batch
        
        clean_batch = False

        # Accessing this if statement produces clean data. 
        if random.random() < self.clean_data_ratio:
             #Do it in a list to make it look like a "batch"
             noise = [None]
             source_rir = [None]
             noise_rirs = [None]
             snr = [None]
             clean_batch = True
        
        # 1. Generate scene
        generated_scene = generate_scenes_batch.generate_scene(
            source_rir=source_rir, noise_rirs=noise_rirs,
            source=audio, noise=noise, snr=snr, sr=self.sr
        )

        # Stack the generated scene on the channel axis, this means that we did not augment our audio, and we are using the one-channel audio.
        if clean_batch:
            # If we are utilizing ambisonics, then stack the one-channel audio on the 4 channels, the feature extraction will give the Intesity Vectors
            if self.in_channels == 7:
                generated_scene = repeat(generated_scene, "B C L_full -> B (repeat C) L_full", repeat = 4)
            # If we are utilizing binaural training, then stack them on left and right channels.
            elif self.in_channels == 2:
                generated_scene = repeat(generated_scene, "B C L_full -> B (repeat C) L_full", repeat = 2)
            elif self.in_channels == 1:
                generated_scene = generated_scene
            else:
                raise Exception(f"Unknown channel count {self.in_channels}")

        # 2. Create Mel Spectrogram and pad
        fbank = self._wav2fbank(generated_scene)
        fbank = pad_or_truncate_batch(fbank, self.input_length) # B, C, T, F
        B, C, T, F_mel = fbank.shape

        # Generate all random start indices at once
        rand_starts = torch.randint(
            0, T - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device
        )

        # Create indices for gathering
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)
        
        # Expand fbank and indices for gathering
        # THIS IS THE FIX: Expand fbank to match the number of samples dimension
        fbank_expanded = fbank.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1, -1)
        indices_expanded = indices.view(B, self.nr_samples_per_audio, 1, self.target_length, 1).expand(-1, -1, C, -1, F_mel)
        
        # Use gather to select all windows at once. The shapes now match correctly.
        return_fbank = torch.gather(fbank_expanded, 3, indices_expanded)

        # 4. Flatten, shuffle, and cast (no change)
        flattened = collate_fn(return_fbank.to(torch.bfloat16)) # B*N, C, L, F
        idx = torch.randperm(flattened.size(0))
        
        return flattened[idx, ...], collate_fn(mask)    

    def log_first_spectrogram(self, patches, title, loss, **kwargs):
        sample_img = patches[:1].clone()
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = torch.min(patches_unflattened), torch.max(patches_unflattened)
        combined = self.patch_strategy.combine_patches(patches_unflattened, self.input_shape).detach().cpu().float().numpy()
        combined =  combined.transpose(0, 1, 3, 2)
        title_plot = "Azimuth: {} Elevation: {}".format(kwargs["direction"][0], kwargs["direction"][1]) if "direction" in kwargs else f"Loss: {loss}"
        fig = plot_fbank(combined[0],
                            vmin = color_min,
                            vmax = color_max,
                            title = title_plot)
        self.logger.experiment.add_figure(f'{title}', fig, global_step = self.global_step)
   
    def log_first_spectrogram_with_mask(self, patches, mask, title):
        sample_img = patches[:1].clone().detach().cpu()
        # Selecting using boolean indexing
        sample_img[:, mask[0], :] = 0
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = torch.min(patches_unflattened), torch.max(patches_unflattened)
        combined = self.patch_strategy.combine_patches(patches_unflattened, self.input_shape).cpu().float().numpy()
        combined =  combined.transpose(0, 1, 3, 2)
        fig = plot_fbank(combined[0],
                            vmin = color_min,
                            vmax = color_max)
        self.logger.experiment.add_figure(f'{title}', fig, global_step = self.global_step)

    def loss(self, pred, target, mask):
        """
        Mask is the boolean tensor containing 1 for masked indices, and 0 for non-mask-indices
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask.int().float()).sum() / mask.int().float().sum()  # mean loss on masked patches
        loss = loss * self.in_channels #Weight the loss w.r.t in channels!
        return loss
    
    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        # Just making sure that they are in training mode
        audio_input, mask = self._prepare_batch(batch)
        x, patches, loss = self.forward(audio_input, mask)

        if self.global_step % self.log_every_n_steps == 0:
            self.log_first_spectrogram(
                patches, title="input_spectrogram", loss=loss
            )
            self.log_first_spectrogram_with_mask(
                patches, ~mask, title="target_spectrogram"
            )
            self.log_first_spectrogram_with_mask(
                patches, mask, title="masked_spectrogram"
            )
            self.log_first_spectrogram(
                x, title="generated_spectrogram", loss = loss
            )

        self.log_dict(
            {
                "MSE_Loss": loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        audio_input, mask = batch
        _, _, loss = self.forward(audio_input)
        self.log_dict(
            {
                "Validation_MSE_Loss": loss,
            }
        )
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        audio_trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = None
        if self.trainer_name == "adamW":
            optimizer = torch.optim.AdamW(
                audio_trainables,
                self.lr,
                weight_decay=self.weight_decay,
                betas=(self.b1, self.b2),
            )
        cosine_annealing = transformers.get_cosine_schedule_with_warmup(optimizer,
                                 num_warmup_steps=10000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}

    def forward(self, x, mask):
        """X is expected to be in B,C,T,F if normal TAR module
        # Here we are actually doing frequency first
        # When we do the forward, it is indeed frequency first!
           Otherwise 1,B,C,T,F ..
        """
        assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
        B = x.shape[0]
        x = x.transpose(2, 3)
        x = self.spectrogram_normalize(x)
        # 1. Patch the input X, we use these later to mask them.
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        # 2. Encode downsampled input
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        x = self.pass_through_encoder(encoded_patches, ~mask, B)
        x = self.pass_through_decoder(x, ~mask, B)
        # Calculate loss on the masked patches
        loss = self.loss(x, patches, mask)
        return x, patches, loss


    def get_audio_representation(self, x, strategy="mean"):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)
        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder(encoded_patches, ~mask, B)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            # We have 25 time patches in 2 second audio. We need to have 20 for STARSS22.
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")

    def get_audio_representation_from_layer(self, x, strategy="mean", block_num=-1):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)

        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder_until_block(encoded_patches, ~mask, B, block_num)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")