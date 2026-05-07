from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # Text
    text_model_name: str = "distilbert-base-uncased"
    text_freeze_base: bool = True
    d_text: int = 256

    # Audio
    audio_model_name: str = "openai/whisper-small"
    audio_freeze_base: bool = True
    audio_input_sr: int = 22050    # sample rate of incoming waveform
    d_audio: int = 256

    # Vision
    vision_n_keyframes: int = 8
    vision_freeze_base: bool = True
    d_vision: int = 256


@dataclass
class AlignmentConfig:
    d_shared: int = 256
    use_attention_fusion: bool = False   # False = mean fusion, True = self-attention


@dataclass
class VAEConfig:
    d_latent: int = 64
    encoder_hidden_dims: tuple = (256, 128)
    beta: float = 1.0                    # KL weight; increase to tighten bottleneck


@dataclass
class ServerConfig:
    # Which LLM backbone to use — see SCENARIO_REGISTRY in emma/server/pipeline.py
    scenario: str = "plain_llm"      # "llava" | "qwen_audio" | "plain_llm"

    # Number of soft tokens injected into the LLM (more = richer context, slower prefill)
    n_soft_tokens: int = 8

    # Multi-task: both heads share the same LLM backbone and projection
    # sentiment_weight / emotion_weight control the combined loss balance
    sentiment_weight: float = 1.0   # weight for MSE regression loss
    emotion_weight: float = 1.0     # weight for BCE classification loss

    # Freeze LLM backbone and only train projection + task heads + VAE decoder
    freeze_llm: bool = True

    # 8-bit quantization via bitsandbytes (reduces GPU memory ~4x; requires bitsandbytes)
    load_in_8bit: bool = False


@dataclass
class EMMAConfig:
    encoder: EncoderConfig = None
    alignment: AlignmentConfig = None
    vae: VAEConfig = None
    server: ServerConfig = None
    use_compression: bool = True    # False = skip VAE, send fused embedding directly

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()
        if self.alignment is None:
            self.alignment = AlignmentConfig()
        if self.vae is None:
            self.vae = VAEConfig()
        if self.server is None:
            self.server = ServerConfig()
