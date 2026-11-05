"""Set the panel's OptiScaler path; does not modify or install game files."""
from pathlib import Path
import argparse
import configparser
import json

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--game-bin', type=Path, required=True)
    p.add_argument('--vk102-manifest', type=Path,
                   help='Register a separately built VK102 package')
    p.add_argument('--small-manifest', type=Path,
                   help='Register a separately built Small package')
    args = p.parse_args()
    target = args.game_bin.resolve()
    if not (target/'Cyberpunk2077.exe').is_file() or not (target/'OptiScaler.ini').is_file():
        p.error('--game-bin must contain Cyberpunk2077.exe and OptiScaler.ini')
    package = args.package.resolve()
    path = package/'panel/panel-config.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    primary = data.get('backend', 'small')
    backends = dict(data.get('backends', {}))
    if data.get('vk102_manifest'):
        backends['102'] = data.pop('vk102_manifest')
    for key, manifest in (('small', args.small_manifest), ('102', args.vk102_manifest)):
        if manifest is None:
            continue
        manifest = manifest.resolve()
        if not (manifest/'port_network_runtime.txt').is_file() or not any(
                (manifest.parent/name).is_file() for name in ('bin/nrvk_bridge.dll', 'nrvk_bridge.dll')):
            p.error('The registered package requires its runtime plan and matching component')
        if key == primary and manifest != package/'manifest':
            p.error('The boot backend must use this package; register the other model instead')
        backends[key] = str(manifest)
    backends[primary] = str(package/'manifest')
    config = package/'config'
    data.update({'ini': str(config/'nr_panel.ini'), 'state': str(config/'panel_state.json'),
                 'manifest': str(package/'manifest')})
    data['optiscaler_ini'] = str(target/'OptiScaler.ini')
    data['backend'] = primary
    data['backends'] = backends
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    controls = configparser.ConfigParser()
    controls.read(config/'nr_panel.ini', encoding='utf-8')
    if not controls.has_section('Backends'):
        controls.add_section('Backends')
    for key, manifest in backends.items():
        controls.set('Backends', key, manifest)
    with (config/'nr_panel.ini').open('w', encoding='utf-8') as handle:
        controls.write(handle)
    (package/'game-files/plugins/nr_bridge.ini').write_text(
        '[bridge]\nbackend=' + primary + '\nmanifest_dir=' + (package/'manifest').as_posix() +
        '\nstatus_json=' + (config/'component_status.json').as_posix() +
        '\npanel_ini=' + (config/'nr_panel.ini').as_posix() +
        '\npanel_state=' + (config/'panel_state.json').as_posix() +
        '\ntail_profile=0\n', encoding='utf-8')
    print('Shared paths configured. Copy game-files contents into bin/x64 after backing up existing files. Recopy nr_bridge.ini after moving this package.')
