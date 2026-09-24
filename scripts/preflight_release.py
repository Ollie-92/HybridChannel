"""Check only Git's explicit candidate file set before a local release review.

Findings require review; absence of findings is not proof of license clearance
or absence of all possible secrets. No files are changed by this checker.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['git','ls-files','-z'],cwd=root,capture_output=True,check=True)
    paths=[s for s in result.stdout.decode().split('\0') if s]
    if not paths:raise SystemExit('No tracked/staged candidate files to review')
    forbidden={'runs','data/external','private_audit','private_data','.venv','venv','__pycache__','.git'}
    signatures=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
                re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
                re.compile(r'(?i)(?:api_key|password|access_token)\s*=\s*[\x27\x22][^\x27\x22\n]{8,}[\x27\x22]')]
    findings=[]
    for rel in paths:
        path=root/rel
        if any(rel==prefix or rel.startswith(prefix+'/') for prefix in forbidden):
            findings.append({'path':rel,'kind':'excluded-content'})
        if path.is_symlink():findings.append({'path':rel,'kind':'symlink'})
        if path.stat().st_size>2_000_000:findings.append({'path':rel,'kind':'large-file'})
        text=path.read_text(encoding='utf-8',errors='replace')
        for i,line in enumerate(text.splitlines(),1):
            if any(pattern.search(line) for pattern in signatures):
                findings.append({'path':rel,'line':i,'kind':'possible-secret'})
            # Match concrete personal roots, without printing their contents.
            if re.search(r'(?:/home/[^ /\x27\x22]+/|[A-Z]:\\Users\\[^\\]+\\)',line):
                findings.append({'path':rel,'line':i,'kind':'personal-absolute-path'})
    report={'candidate_files':len(paths),'paths':paths,'findings':findings,
            'license_clearance':'manual release gate; see docs/license_audit.md',
            'scope':'tracked/staged files only; no claims about hidden credentials outside Git'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'candidate_files':len(paths),'findings':len(findings)}))
    return bool(findings)


if __name__=='__main__':raise SystemExit(main())
