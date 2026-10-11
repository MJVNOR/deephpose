"""Download the official EV-M and TV-M fine-tuned checkpoints into ignored data/."""
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(__file__).resolve().parents[2]
for filename in ('efficientsam3_efficientvit.pt','efficientsam3_tinyvit.pt'):
    print(hf_hub_download('Simon7108528/EfficientSAM3',f'efficientsam3_ft/{filename}',
                          revision='85b05896928f974e308f889d7ccb2eefc069de98',
                          local_dir=root/'data/models/EfficientSAM3'))
