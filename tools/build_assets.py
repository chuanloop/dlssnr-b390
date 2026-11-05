"""Compile the frozen Small/VK102 shaders and prepare relocatable game assets."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def instructions(data):
    words = struct.unpack('<%dI' % (len(data) // 4), data)
    result = []
    pos = 5
    while pos < len(words):
        count = words[pos] >> 16
        if not count or pos + count > len(words):
            raise ValueError('Invalid compiled SPIR-V')
        result.append(words[pos:pos + count])
        pos += count
    return words[:5], result


def compile_shader(compiler, recipe, output):
    source = ROOT / recipe['source']
    subprocess.run([str(compiler), '-V', '--target-env', 'vulkan1.3',
                    *recipe.get('defines', []), '-I' + str(source.parent),
                    '-o', str(output), str(source)], check=True)
    raw = output.read_bytes()
    if 'raw_sha256' in recipe:
        if digest(raw) != recipe['raw_sha256']:
            raise ValueError(f'Compiler/source identity mismatch: {source.name}; use glslang 16.5.0')
        # The insertion positions are bound to the exact raw SHA above. No
        # historical ID is applied to a different compiler output.
        header, ops = instructions(raw)
        ops = [op for op in ops if not (op[0] & 65535 == 71 and len(op) == 3 and op[2] == 42)]
        for entry in recipe['precision_decorations']:
            ops.insert(entry['position'], tuple(entry['words']))
        words = list(header) + [word for op in ops for word in op]
        final = struct.pack('<%dI' % len(words), *words)
        if digest(final) != recipe['sha256']:
            raise ValueError(f'Frozen precision/output mismatch: {source.name}')
        output.write_bytes(final)
    return {'bytes': output.stat().st_size, 'sha256': digest(output.read_bytes())}


def prepare(model, compiler):
    model = model.resolve()
    manifest = model / 'manifest'
    cold = json.loads((manifest / 'cold_resources_index.json').read_text(encoding='utf-8'))
    profile_key = json.loads((model / 'model/extraction-receipt.json').read_text(encoding='utf-8'))['profile']
    profile = json.loads((ROOT / 'tools/model_profiles.json').read_text(encoding='utf-8'))[profile_key]
    network = ROOT / profile['network_dir']
    recipes = json.loads((network / 'recipes.json').read_text(encoding='utf-8'))['shaders']
    if (network / 'cold-hashes.json').is_file():
        expected = json.loads((network / 'cold-hashes.json').read_text(encoding='utf-8'))
        if set(expected) != set(cold) or any(cold[k]['sha256'] != v['sha256'] or cold[k]['bytes'] != v['bytes'] for k, v in expected.items()):
            raise ValueError('Cold packing differs from the frozen model profile')
    spv = model / 'assets/spv'
    spv.mkdir(parents=True, exist_ok=True)
    shaders = {}
    for name, recipe in recipes.items():
        output = spv / (recipe['sha256'] + '__' + name)
        record = compile_shader(compiler, recipe, output)
        record['path'] = '../assets/spv/' + output.name
        shaders[name] = record
    bridge = {}
    for name in ('gather', 'scatter', 'scatter32'):
        filename = 'port_network_bridge_' + name + '.spv'
        output = spv / filename
        record = compile_shader(compiler, {'source': 'source/vulkan/' + filename.replace('.spv', '.comp')}, output)
        record['path'] = '../assets/spv/' + filename
        bridge[name] = record
    template = (network / 'runtime.template.txt').read_text(encoding='utf-8')
    for name, record in cold.items():
        file = Path(record['path'])
        if not file.is_file():
            file = model / 'assets/cold' / file.name
        if file.stat().st_size != record['bytes'] or digest(file.read_bytes()) != record['sha256']:
            raise ValueError('Cold resource integrity: ' + name)
        template = template.replace('@COLD/' + name + '@', '../assets/cold/' + file.name)
    for name, record in shaders.items():
        template = template.replace('@SPV/' + name + '@', record['path'])
    if re.search(r'@(COLD|SPV)/', template):
        raise ValueError('Unresolved runtime asset')
    runtime = manifest / 'port_network_runtime.txt'
    runtime.write_text(template, encoding='utf-8', newline='\n')
    data = {
        'schema': 'intel-dlss5nr-game-1', 'model': profile['name'],
        'profile': profile_key, 'backend_key': profile['backend_key'],
        'rows': profile['rows'], 'buffers': profile['buffers'], 'passes': profile['passes'], 'outputs': 1, 'history_links': 2,
        'input': [1280, 720], 'network_shaders': shaders, 'bridge_shaders': bridge,
        'cold': {name: {'path': '../assets/cold/' + Path(v['path']).name,
                        'bytes': v['bytes'], 'sha256': v['sha256']} for name, v in cold.items()},
        'image_inputs': 'zero-initialized, filled from game images/controls/history; no fixture files',
    }
    (manifest / 'port_network_manifest.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    generated = model / 'generated'
    generated.mkdir(exist_ok=True)
    generated.joinpath('port_network_bridge_backend_model_identity.inc').write_text(
        f'static constexpr size_t kPlanRows={profile["rows"]},kPlanBuffers={profile["buffers"]},kPlanPasses={profile["passes"]},kPlanOutputs=1,kPlanHistory=2;\n'
        'static constexpr const char* kPreImageName="ioport_0_ioport_images";\n'
        f'static constexpr const char* kPostImageName="ioport_{profile["post_block"]}_ioport_images";\n', encoding='utf-8')
    manifest_sha = digest((manifest / 'port_network_manifest.json').read_bytes())
    runtime_sha = digest(runtime.read_bytes())
    lines = ['struct FrozenFile { const char* path; const char* sha; };',
             f'static constexpr const char* frozen_manifest_sha="{manifest_sha}";',
             f'static constexpr const char* frozen_runtime_sha="{runtime_sha}";',
             'static constexpr FrozenFile frozen_files[]={']
    for record in list(data['cold'].values()) + list(shaders.values()) + list(bridge.values()):
        lines.append(json.dumps(record['path']) + ',' + json.dumps(record['sha256']))
    lines = [line if not line.startswith('"') else '{' + line + '},' for line in lines]
    lines.append('};')
    for name, record in bridge.items():
        lines.append(f'static constexpr const char* {name}_spv=' + json.dumps(record['path']) + ';')
    generated.joinpath('port_network_bridge_backend_frozen.inc').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    generated.joinpath('nrvk_build_identity.inc').write_text(
        f'static constexpr const char* kFrozenBuildId="intel-dlss5nr-{profile_key}-source";\n'
        f'static constexpr const char* kFrozenManifestSha="{manifest_sha}";\n'
        f'static constexpr const char* kFrozenRuntimeSha="{runtime_sha}";\n', encoding='utf-8')
    print(json.dumps({'status': 'GAME_ASSETS_PREPARED', 'network_shaders': len(shaders),
                      'cold_resources': len(cold), 'manifest_sha256': manifest_sha,
                      'runtime_sha256': runtime_sha, 'developer_samples': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--glslang', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.glslang.resolve())
