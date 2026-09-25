"""Apply a backed-up, idempotent integration to this project's existing UI."""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import re

MARKER = '# ADC_REST_PERSISTENCE_V1'
BLOCK = '''
        # ADC_REST_PERSISTENCE_V1
        # Save the retry ID before the network call. JSON is already on disk.
        import uuid
        run_id = str(uuid.uuid4())
        output.with_suffix(".run_id.txt").write_text(run_id, encoding="utf-8")
        try:
            saved = DataClient().save_run(asdict(state), run_id=run_id)
        except Exception as exc:
            message = f"Shared DB save failed; JSON retained. Retry run ID: {run_id}\\n{exc}\\n"
            self.after(0, lambda message=message: self._log(message))
        else:
            message = f"Shared Qdrant save complete. Run ID: {saved['run_id']}\\n"
            self.after(0, lambda message=message: self._log(message))
'''

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
path = args.project_root / 'src/agent1_orchestrator/ui.py'
source = path.read_text(encoding='utf-8-sig')
if MARKER in source:
    print('REST integration is already present; no changes made.')
    raise SystemExit(0)
anchor = 'from agents.orchestrator import OrchestratorAgent'
if anchor not in source:
    raise SystemExit('Unsupported UI imports; original file unchanged. Use README integration notes.')
source = source.replace(anchor, '''# Make direct UI launch resolve both project and Agent 1 imports.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src" / "agent1_orchestrator"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from adc_shared.client import DataClient
''' + anchor, 1)
source = source.replace('self.project_root = Path(__file__).resolve().parent\n',
                        'self.project_root = Path(__file__).resolve().parents[2]\n')
match = re.search(r'        agent = OrchestratorAgent\(.*?\n        \)', source, re.S)
if not match:
    raise SystemExit('Constructor not recognized; original file unchanged.')
constructor = match.group()
for key in ('enable_a2a', 'populate_vector_db'):
    if re.search(rf'\b{key}\s*=', constructor):
        constructor = re.sub(rf'\b{key}\s*=\s*[^,\n]+', key + '=False', constructor)
    else:
        constructor = constructor[:-len('        )')] + f'            {key}=False,\n        )'
source = source[:match.start()] + constructor + source[match.end():]
anchor = '        output.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")\n'
if source.count(anchor) != 1:
    raise SystemExit('JSON output statement not recognized; original file unchanged.')
source = source.replace(anchor, anchor + BLOCK, 1)
ast.parse(source)
backup = path.with_name(path.name + '.before_rest_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
backup.write_bytes(path.read_bytes())
path.write_text(source, encoding='utf-8')
print(f'Updated: {path}\nBackup: {backup}\nAgent 2 and old embedded Qdrant indexing remain disabled.')
