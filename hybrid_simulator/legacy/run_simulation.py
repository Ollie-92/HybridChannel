"""Run a satellite channel simulation using a JSON configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent


def sha256(path):
    """Hash the exact bytes of one input without modifying it."""
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record_scene(scene, run):
    """Record XML and referenced mesh bytes; reject missing or unsafe inputs."""
    inputs = [scene]
    for element in ET.parse(scene).iter('string'):
        if element.get('name') == 'filename':
            target = (scene.parent / element.attrib['value']).resolve()
            if not target.is_relative_to(scene.parent):
                raise ValueError('Scene assets must remain inside the supplied scene directory')
            if target.is_symlink() or not target.is_file():
                raise ValueError(f'Missing/linked scene asset: {target.name}')
            inputs.append(target)
    records = [{'path': str(p.relative_to(scene.parent)), 'sha256': sha256(p),
                'bytes': p.stat().st_size} for p in sorted(set(inputs))]
    (run / 'data_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


def make_test_scene(path):
    """Create a small ground plane for integration testing, not a city experiment."""
    path.write_text('''<scene version="3.0.0">
  <bsdf type="itu-radio-material" id="ground_material">
    <string name="type" value="concrete"/>
  </bsdf>
  <shape type="rectangle" id="test_ground">
    <transform name="to_world"><scale x="100" y="100" z="1"/></transform>
    <ref id="ground_material" name="bsdf"/>
  </shape>
</scene>
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=REPO / 'Result')
    parser.add_argument('--scene', type=Path, help='Privately supplied scene, overriding config')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    allowed = {'experiment_id', 'description', 'scene', 'cli', 'blocks', 'paper_figure'}
    if set(config) - allowed:
        raise ValueError('Unrecognized configuration fields: ' + str(set(config) - allowed))
    eid = config['experiment_id']
    if not eid.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid experiment ID')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / (eid + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                  + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    scene_value = args.scene if args.scene else config['scene']
    if scene_value == 'generated-test-ground':
        scene = run / 'test_ground.xml'
        make_test_scene(scene)
    else:
        scene = Path(scene_value)
        scene = (scene if scene.is_absolute() else REPO / scene).resolve()
        if not scene.is_file():
            raise FileNotFoundError('Supply the scene XML with --scene; see README.md')
    record_scene(scene, run)
    cli = config['cli']
    permitted = {'rx-x','rx-y','rx-z','fc-ghz','tx-power-dbm','altitude-m','initial-elev-deg',
                 'speed-mps','heading-deg','duration-s','num-steps','samples-per-src','max-depth',
                 'max-num-paths','rt-pre-filter-db','num-ant','seed','consistent-random-clusters'}
    if set(cli) - permitted:
        raise ValueError('Unsupported scientific option: ' + str(set(cli) - permitted))
    command = [sys.executable, '-m', 'Hybrid_channel.trajectory', '--scene', str(scene),
               '--output', str(run / 'channel.npz'), '--summary', str(run / 'summary.txt')]
    for name, value in cli.items():
        if isinstance(value, bool):
            if value: command.append('--' + name)
        else:
            command.extend(['--' + name, str(value)])
    env = os.environ.copy()
    # Resolve the local channel package while the solver runs in its output directory.
    env['PYTHONPATH'] = str(REPO)
    # The original package/global environment never supplies an implicit cache.
    for key, sub in {'XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'cache/matplotlib',
                     'TMPDIR':'tmp', 'HOME':'home'}.items():
        directory = run / sub
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLBACKEND'] = 'Agg'
    git = subprocess.run(['git','rev-parse','HEAD'], cwd=REPO, capture_output=True, text=True)
    status = subprocess.run(['git','status','--porcelain'], cwd=REPO, capture_output=True, text=True)
    snapshot = dict(config, resolved_scene=str(scene), command=command,
                    created_utc=datetime.now(timezone.utc).isoformat())
    (run / 'config.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    metadata = {'python':sys.version, 'platform':platform.platform(),
                'git_commit':git.stdout.strip() if git.returncode == 0 else None,
                'git_dirty':bool(status.stdout.strip()) if status.returncode == 0 else None,
                'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                'package_sha256':{p.name:sha256(p) for p in (REPO/'Hybrid_channel').glob('*.py')},
                'status':'running'}
    metadata_path = run / 'environment.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('RUN_DIR=' + str(run), flush=True)
    with (run / 'run.log').open('w') as log:
        result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata['returncode'] = result.returncode
    metadata['status'] = 'passed' if result.returncode == 0 else 'failed'
    if result.returncode == 0:
        import numpy as np
        with np.load(run/'channel.npz', allow_pickle=False) as data:
            metadata['valid_steps'] = len(data['valid_steps'])
        if metadata['valid_steps'] != int(cli['num-steps']):
            metadata['status'] = 'partial'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('STATUS=' + metadata['status'])
    return result.returncode or (2 if metadata['status'] != 'passed' else 0)


if __name__ == '__main__':
    raise SystemExit(main())
