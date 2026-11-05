"""Assemble locally built files; never installs into the game."""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def assemble(model, output):
    model, output = model.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Package output must be new or empty')
    metadata = json.loads((model/'manifest/port_network_manifest.json').read_text(encoding='utf-8'))
    backend = metadata['backend_key']
    for name in ('manifest', 'assets/cold', 'assets/spv'):
        shutil.copytree(model / name, output / name)
    shutil.copytree(ROOT / 'panel', output / 'panel', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config = output / 'config'
    config.mkdir()
    plugins = output / 'game-files/plugins'
    plugins.mkdir(parents=True)
    shutil.copy2(ROOT / 'build/bridge/Release/CyberpunkNRBridge.asi', plugins)
    shutil.copy2(ROOT / 'source/bridge/nr_hdr_proxy.hlsl', plugins)
    (plugins / 'NRB-deferred-identity.enable').write_text('enabled\n')
    # `build/vulkan/Release` is the old CMake tree and stopped being rebuilt when the vk102
    # profile landed; `build/vulkan-vk102` is the one the runner is built from. Prefer the
    # newer of the two instead of trusting a hardcoded path -- the previous form shipped a
    # 2026-10-31 DLL while the tree already had a newer backend.
    # The bridge is per profile: each build is compiled against its own frozen plan identity
    # (rows/buffers/passes) and refuses a plan that is not its own, so shipping the full
    # profile's DLL inside a Small package would fail at init, not at packaging time.
    # `local/<profile>` is the model dir and `build/vulkan-<profile>` is the CMake dir for it
    # (`vk102`, `small`). The two legacy locations stay as a fallback for vk102 only.
    profile_dir = model.name
    bridge = ROOT / f'build/vulkan-{profile_dir}/nrvk_bridge.dll'
    if not bridge.is_file():
        if profile_dir != 'vk102':
            raise FileNotFoundError(
                f'nrvk_bridge.dll for profile {profile_dir!r} not built ({bridge} missing); '
                f'the full profile is not a substitute')
        bridge = max((p for p in (ROOT / 'build/vulkan-vk102/nrvk_bridge.dll',
                                  ROOT / 'build/vulkan/Release/nrvk_bridge.dll') if p.is_file()),
                     key=lambda p: p.stat().st_mtime, default=None)
    if bridge is None:
        raise FileNotFoundError('nrvk_bridge.dll not built (build/vulkan-vk102 or build/vulkan)')
    shutil.copy2(bridge, output / 'game-files/nrvk_bridge.dll')
    (output/'bin').mkdir()
    shutil.copy2(bridge, output/'bin/nrvk_bridge.dll')
    # AMD's control defaults (nr_graph.cpp:2364-2374, adopted in M1-T39): tone 1, structure 1,
    # skin -1 ("fall back to structure"), auto mask on. The previous template wrote tone=0 and
    # skin=0, i.e. the pre-M1-T39 configuration, which measured 0.4271 dB worse -- a fresh
    # package would have shipped that regression no matter how current the shaders were.
    (config / 'nr_panel.ini').write_text('[NR]\nbackend=' + backend + '\nenabled=1\nstyle=0\nintensity=1.0\nlocal_structure=1.0\nlocal_tone=1\nskin_structure=-1\nauto_mask=1\ndisplay_strength=1.0\n\n[Backends]\n' + backend + '=' + (output/'manifest').as_posix() + '\n', encoding='utf-8')
    (plugins / 'nr_bridge.ini').write_text(
        '[bridge]\nbackend=' + backend + '\nmanifest_dir=' + (output/'manifest').as_posix() +
        '\nstatus_json=' + (config/'component_status.json').as_posix() +
        '\npanel_ini=' + (config/'nr_panel.ini').as_posix() +
        '\npanel_state=' + (config/'panel_state.json').as_posix() +
        '\ntail_profile=0\n', encoding='utf-8')
    panel = {'backend': backend, 'backends': {backend: str(output/'manifest')},
             'ini': str(config/'nr_panel.ini'), 'state': str(config/'panel_state.json'),
             'manifest': str(output/'manifest'), 'optiscaler_ini': ''}
    (output/'panel/panel-config.json').write_text(json.dumps(panel, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(ROOT/'panel/Launch-NR-Panel-Game.cmd', output/'Launch-NR-Panel-Game.cmd')
    (output/'Configure-Game.cmd').write_text('@echo off\npy -3 "%~dp0panel\\configure_game.py" --package "%~dp0." --game-bin "%~1"\npause\n', encoding='ascii')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.model, args.out)
