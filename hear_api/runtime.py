import sys
sys.path.append('..')
import torch
from src.model import GRAMM
from src.patching import PatchStrategy
from .feature_helper_gram_m import FeatureExtractor, get_timestamps

configs = {
    "tiny": {
        "depth": 12, "num_heads": 3, "embed_dim": 192
    },
    "small": {
        "depth": 12, "num_heads": 6, "embed_dim": 384
    },
    "medium": {
        "depth": 12, "num_heads": 8, "embed_dim": 512
    },
    "base": {
        "encoder_depth": 24, 
        "encoder_embed_dim": 768,
        "decoder_depth": 8, 
        "decoder_num_heads": 8, 
        "decoder_embed_dim": 512,
        
    },
    "large": {
        "depth": 24, "num_heads": 16, "embed_dim": 1024
    },
    "huge": {
        "depth": 32, "num_heads": 16, "embed_dim": 1280
    }
}

class RuntimeGRAMM(torch.nn.Module):
    def __init__(self, 
                 model_size,
                 config, 
                 in_channels,
                 weights,
                 fshape, 
                 tshape,
                 fstride, 
                 tstride,
                 input_tdim,
                 strategy: str = "raw",
                 layer: int = None,
                 **kwargs) -> None:
        super().__init__()
        encoder_depth, encoder_embed_dim, decoder_depth, decoder_num_heads, decoder_embed_dim = configs[model_size].values()
        self.decoder_window_sizes = kwargs.get("decoder_window_sizes", [2,5,10,25,50,100,0,0])
        self.use_mwmae_decoder = kwargs.get("use_mwmae_decoder", False)
        self.num_mel_bins = kwargs.get("num_mel_bins", 128)
        self.in_channels = kwargs.get("in_channels", 2)
        self.model = GRAMM(
            patch_strategy=PatchStrategy(
                input_tdim=input_tdim,
                input_fdim=self.num_mel_bins,
                tstride=tstride,
                tshape=tshape,
                fstride=fstride,
                fshape=fshape,
            ),
            in_channels=in_channels,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,           
            decoder_num_heads=decoder_num_heads,
            encoder_embedding_dim=encoder_embed_dim,
            decoder_embedding_dim=decoder_embed_dim,
            mlp_ratio=config.model.mlp_ratio,
            ssm_cfg={"d_state":config.model.d_state,
                    "d_conv":config.model.d_conv,
                    "expand":config.model.expand},
            rms_norm = config.model.rms_norm,
            residual_in_fp32 = config.model.residual_in_fp32,
            fused_add_norm = config.model.fused_add_norm,
            precision = config.trainer.precision,
            norm_epsilon = config.model.norm_epsilon,
            use_mwmae_decoder=self.use_mwmae_decoder,
            decoder_window_sizes = self.decoder_window_sizes,
        )
        self.model.load_state_dict(weights["state_dict"])
        self.model.eval()
        # The input size to the model is the input_t_dim and the number of mel bins.
        self.grid_size = self.model.grid_size
        self.input_size = (input_tdim, self.num_mel_bins)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.model.encoder_embedding_dim
        self.timestamp_embedding_size = self.model.encoder_embedding_dim

        # That's where we set the sample rate!
        self.sample_rate = 32000
        self.strategy = strategy
        self.mel_spec = FeatureExtractor(in_channels=self.in_channels,
                                         sr = self.sample_rate,
                                         num_mel_bins = self.num_mel_bins) 
        self.until_layer = layer 
    
    def to_feature(self, batch_audio):
        return self.mel_spec(batch_audio)
    
    def encode(self, x):
        unit_frames = self.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with reflect mode
            pad_arg = (0, 0, 0, pad_frames)  # (channel, channel, height, height, width, width)
            x = torch.nn.functional.pad(x, pad_arg, mode="constant")
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i*unit_frames:(i+1)*unit_frames, :]
            with torch.no_grad():
                if self.until_layer is not None:
                    embedding = self.model.get_audio_representation_from_layer(x_inp, strategy=self.strategy, block_num = self.until_layer)
                else:
                    embedding = self.model.get_audio_representation(x_inp, strategy=self.strategy)
            embeddings.append(embedding)
        # Stack the embeddings here if it is raw
        # Also get rid of the padding if it is raw!
        if self.strategy == "raw":
            x = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames] # remove padded tail
            return x
        else:
            x = torch.stack(embeddings, dim=1) 
            return x
    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        return x
    
    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)  
        # This takes the mean embedding across the scene! 
        x = torch.mean(x, dim=1)
        return x
    
    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
    
                                        
class RuntimeSpatialMambaDownSample(torch.nn.Module):
    def __init__(self, 
                 model_size,
                 config, 
                 in_channels,
                 weights,
                 fshape, 
                 tshape,
                 fstride, 
                 tstride,
                 input_tdim,
                 strategy: str = "raw",
                 **kwargs) -> None:
        super().__init__()
        encoder_depth, encoder_embed_dim, decoder_depth, decoder_num_heads, decoder_embed_dim = configs[model_size].values()
        self.decoder_window_sizes = kwargs.get("decoder_window_sizes", [2,5,10,25,50,100,0,0])
        self.use_mwmae_decoder = kwargs.get("use_mwmae_decoder", False)
        self.num_mel_bins = kwargs.get("num_mel_bins", 128)
        self.in_channels = kwargs.get("in_channels", 2)
        self.model = SpatialMambaDownSample(
            patch_strategy=PatchStrategy(
                input_tdim=input_tdim,
                input_fdim=self.num_mel_bins,
                tstride=tstride,
                tshape=tshape,
                fstride=fstride,
                fshape=fshape,
            ),
            in_channels=in_channels,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,           
            decoder_num_heads=decoder_num_heads,
            encoder_embedding_dim=encoder_embed_dim,
            decoder_embedding_dim=decoder_embed_dim,
            mlp_ratio=config.model.mlp_ratio,
            ssm_cfg={"d_state":config.model.d_state,
                    "d_conv":config.model.d_conv,
                    "expand":config.model.expand},
            rms_norm = config.model.rms_norm,
            residual_in_fp32 = config.model.residual_in_fp32,
            fused_add_norm = config.model.fused_add_norm,
            precision = config.trainer.precision,
            norm_epsilon = config.model.norm_epsilon,
            use_mwmae_decoder=self.use_mwmae_decoder,
            decoder_window_sizes = self.decoder_window_sizes,
        )
        self.model.load_state_dict(weights["state_dict"])
        self.model.eval()
        # The input size to the model is the input_t_dim and the number of mel bins.
        self.grid_size = self.model.grid_size
        self.input_size = (input_tdim, self.num_mel_bins)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.model.encoder_embedding_dim
        self.timestamp_embedding_size = self.model.encoder_embedding_dim

        # That's where we set the sample rate!
        self.sample_rate = 32000
        self.strategy = strategy
        self.mel_spec = FeatureExtractor(in_channels=self.in_channels,
                                         sr = self.sample_rate,
                                         num_mel_bins = self.num_mel_bins)  
    
    def to_feature(self, batch_audio):
        return self.mel_spec(batch_audio)
    
    def encode(self, x):
        unit_frames = self.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with reflect mode
            pad_arg = (0, 0, 0, pad_frames)  # (channel, channel, height, height, width, width)
            x = torch.nn.functional.pad(x, pad_arg, mode="constant")
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i*unit_frames:(i+1)*unit_frames, :]
            with torch.no_grad():
                embedding = self.model.get_audio_representation(x_inp, strategy=self.strategy)
            embeddings.append(embedding)
        # Stack the embeddings here if it is raw
        # Also get rid of the padding if it is raw!
        if self.strategy == "raw":
            x = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames] # remove padded tail
            return x
        else:
            x = torch.stack(embeddings, dim=1) 
            return x
    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        return x
    
    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)  
        # This takes the mean embedding across the scene! 
        x = torch.mean(x, dim=1)
        return x
    
    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
