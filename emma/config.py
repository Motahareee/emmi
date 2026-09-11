from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # family: "clip" (HF CLIP ViT-B/32) | "mobileclip" (open_clip MobileCLIP2-S0)
    family: str = "clip"
    # Both text and image encoders use the same CLIP checkpoint (family="clip")
    model_name: str = "openai/clip-vit-base-patch32"
    text_freeze_base:  bool = True
    image_freeze_base: bool = True


@dataclass
class AlignmentConfig:
    d_shared: int = 512     # joint embedding dimension (512 for both encoder families)
    fusion:   str = "mean"  # "mean" | "concat" | "match" — see CrossModalAlignment


@dataclass
class VAEConfig:
    d_latent: int = 64
    encoder_hidden_dims: tuple = (256, 128)
    beta: float = 1.0          # KL weight; increase to tighten bottleneck


@dataclass
class ServerConfig:
    scenario: str = "plain_llm"    # "llava" | "qwen_audio" | "plain_llm"
    n_soft_tokens: int = 8
    freeze_llm: bool = True
    load_in_8bit: bool = False


@dataclass
class EMMAConfig:
    encoder:  EncoderConfig  = None
    alignment: AlignmentConfig = None
    vae:      VAEConfig      = None
    server:   ServerConfig   = None
    use_compression: bool = True

    def __post_init__(self):
        if self.encoder   is None: self.encoder   = EncoderConfig()
        if self.alignment is None: self.alignment = AlignmentConfig()
        if self.vae       is None: self.vae       = VAEConfig()
        if self.server    is None: self.server    = ServerConfig()
