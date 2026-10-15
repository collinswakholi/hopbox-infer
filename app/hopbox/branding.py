"""One source of application and installer identity."""
import json
from hopbox.paths import bundle_root

BRANDING = json.loads((bundle_root() / 'hopbox/branding.json').read_text(encoding="utf-8"))
