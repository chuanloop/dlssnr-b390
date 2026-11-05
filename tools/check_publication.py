"""Validate the Git source allowlist and optionally prepare an upload ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.md', '.txt', '.json', '.py', '.ps1', '.cmd', '.cpp', '.h', '.comp', '.glsl', '.hlsl', '.inc', '.sha256', '.yml'}
SPECIAL = {'.gitignore', '.gitattributes', 'LICENSE', 'CMakeLists.txt'}


def check(output, archive):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().strip('\0').split('\0')
    rows = []
    for name in names:
        path = ROOT/name
        if path.suffix.lower() not in ALLOWED and path.name not in SPECIAL:
            raise ValueError('Not an allowed source file: '+name)
        if name.startswith('.github/') and name != '.github/workflows/publish-experimental.yml':
            raise ValueError('Unexpected publication workflow: '+name)
        if name.split('/')[0] not in ('source', 'tools', 'panel', 'third_party', '.github', 'docs') and '/' in name:
            raise ValueError('Unexpected top-level folder: '+name)
        data = path.read_bytes()
        if name in SCREENSHOTS:
            if not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Invalid documentation screenshot: '+name)
            rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            continue
        text = data.decode('utf-8-sig')
        if '\0' in text:
            raise ValueError('Binary payload in source: '+name)
        if re.search(r'(?:E:[/\\]NR-Workspaces|D:[/\\]Codex-NR|G:[/\\]epic|[A-Z]:[/\\]Users[/\\])', text, re.I):
            raise ValueError('Developer path in publication: '+name)
        if re.search(r'github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----', text):
            raise ValueError('Credential-like content in publication: '+name)
        rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    result = {'source_and_documentation_only': True, 'files': len(rows), 'bytes': sum(x['bytes'] for x in rows),
              'documentation_screenshots': sum(x['path'] in SCREENSHOTS for x in rows),
              'weights': 0, 'compiled_shaders': 0, 'native_binaries': 0,
              'game_samples': 0, 'uploaded': False, 'entries': rows}
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if archive:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for row in rows:
                z.write(ROOT/row['path'], 'intel-dlss5nr-cyberpunk2077/'+row['path'])
        result['archive'] = str(archive)
        result['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='entries'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--zip', type=Path)
    args = p.parse_args()
    check(args.out.resolve(), args.zip.resolve() if args.zip else None)
