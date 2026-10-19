"""Parameter-budget profiles for all 15 process families; no trained weights."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VARIANT=json.loads((ROOT/'variant.json').read_text(encoding='utf-8'))['variant']
assert VARIANT in ('small','normal','large'), VARIANT

def choice(small,normal,large):
 return {'small':small,'normal':normal,'large':large}[VARIANT]
