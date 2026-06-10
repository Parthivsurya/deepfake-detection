from .temporal_vit import TemporalViT
from .audio_encoder import AudioEncoder, Wav2VecAudioEncoder, build_audio_encoder
from .av_sync import AVSyncHead
from .cross_attention_fusion import CrossAttentionFusion, CrossAttentionBlock
from .detector import MultimodalDeepfakeDetector
from .optimization import (
    apply_global_unstructured_pruning,
    apply_dynamic_quantization,
    weight_sparsity,
    parameter_count,
)
