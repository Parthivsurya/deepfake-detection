from .base import VideoManifest, VideoClipDataset
from .faceforensics import FaceForensicsBuilder
from .celebdf import CelebDFBuilder
from .dfdc import DFDCBuilder
from .fakeavceleb import FakeAVCelebBuilder

BUILDERS = {
    "faceforensics": FaceForensicsBuilder,
    "celebdf": CelebDFBuilder,
    "dfdc": DFDCBuilder,
    "fakeavceleb": FakeAVCelebBuilder,
}
