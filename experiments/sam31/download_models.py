"""Download pinned SAM 3.1 weights, using local HF credentials or root .env."""
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'daa63191845a41281374e725f4c9e51c7a824460'


def main():
    load_dotenv(ROOT / '.env')
    for filename in ('config.json', 'sam3.1_multiplex.pt'):
        print(hf_hub_download('facebook/sam3.1', filename, revision=REVISION,
                             local_dir=ROOT / 'data/models/SAM31'), flush=True)


if __name__ == '__main__':
    main()
