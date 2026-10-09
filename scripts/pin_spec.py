"""Copy the engine's committed OpenAPI documents into spec/ and record where they came from."""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent.parent / '1. Engine'
FILES = ('meridian-public.json', 'meridian-admin.json')


def git(*args: str) -> bytes:
    return subprocess.run(['git', '-C', str(ENGINE), *args], check=True, capture_output=True).stdout


def main() -> None:
    # [1] Copy byte for byte from engine HEAD, never the working tree
    (ROOT / 'spec').mkdir(exist_ok=True)
    for name in FILES:
        (ROOT / 'spec' / name).write_bytes(git('show', f'HEAD:deploy/openapi/{name}'))

    # [2] Record the commit and API versions
    def version(name: str) -> str:
        return str(json.loads((ROOT / 'spec' / name).read_text(encoding='utf-8'))['info']['version'])

    source = {
        'engineCommit': git('rev-parse', '--short=12', 'HEAD').decode().strip(),
        'pinnedAt': date.today().isoformat(),
        'publicVersion': version('meridian-public.json'),
        'adminVersion': version('meridian-admin.json'),
    }
    (ROOT / 'spec' / 'SOURCE.json').write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(f'Pinned engine spec at {source["engineCommit"]} ({source["publicVersion"]}).')


if __name__ == '__main__':
    main()
