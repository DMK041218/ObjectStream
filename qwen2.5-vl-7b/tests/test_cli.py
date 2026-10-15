import json
from pathlib import Path
import subprocess
import sys


def test_dry_run_without_weights_or_video():
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output([
        sys.executable, str(root / 'scripts/run_qwen25vl_objectstream.py'),
        '--video_path', 'not-a-real-video.mp4', '--question', 'What happens?',
        '--dry_run',
    ], text=True)
    result = json.loads(output)
    assert result['dry_run']
    assert result['qwen_config']['short_frames'] == 8
    assert result['entity_config']['tokens_per_entity'] == 8
