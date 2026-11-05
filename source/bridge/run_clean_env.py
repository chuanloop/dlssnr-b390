# -*- coding: utf-8 -*-
"""Run a build command with a de-duplicated Windows environment.

MSBuild's .NET environment dictionary rejects an env block containing the
same variable name with different letter case (e.g. HTTPS_PROXY and
https_proxy). This wrapper keeps the FIRST occurrence (case-insensitive)
of every variable and drops the rest, then execs the command.
"""
import os
import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("usage: run_clean_env.py <command> [args...]")
        return 2
    seen = {}
    for key, value in os.environ.items():
        low = key.lower()
        if low not in seen:
            seen[low] = (key, value)
    env = {key: value for key, value in seen.values()}
    # Drop proxy variables entirely for the build.
    for low in list(env):
        if low.endswith("_proxy"):
            env.pop(low)
    try:
        return subprocess.call(sys.argv[1:], env=env)
    except FileNotFoundError as exc:
        print("command not found:", exc)
        return 127

if __name__ == "__main__":
    sys.exit(main())
