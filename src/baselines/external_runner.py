"""Run an unmodified upstream trainer while persisting machine-readable status."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from baselines.status import RunStatus


def run_upstream(command:list[str],cwd:Path,run:RunStatus,env:dict[str,str]|None=None)->int:
    log_path=run.run_dir/"train.log"
    process_env=os.environ.copy()
    if env:process_env.update(env)
    with log_path.open("a") as log:
        process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,env=process_env)
        run.update(status="running",stage="upstream_training",pid=process.pid,command=command,cwd=str(cwd),log=str(log_path))
        while True:
            code=process.poll()
            if code is not None:break
            run.update(status="running",stage="upstream_training",pid=process.pid)
            time.sleep(30)
    run.update(status="complete" if code==0 else "failed",stage="upstream_training",return_code=code)
    return int(code)
