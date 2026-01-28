import asyncio, sys
from pathlib import Path

# n = input('Enter number of experiments: ')
while True:
    state = input('Enter state (zero, plus): ')
    if state in ['zero', 'plus']:
        break

filename = "full_prep_sim_" + state + ".py"
parent_dir = "logs_prep_" + state
SCRIPT = Path.cwd() / filename
LOGS = Path(parent_dir)

MAX_PARALLEL = 24
SEMA = asyncio.Semaphore(MAX_PARALLEL)

async def run_exp(s, p):
    p_str =  f"p{str(p).split('.')[1]}"
    async with SEMA:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(SCRIPT), str(s), str(p), 
            stdout=open(LOGS / p_str / f"{str(s)}.log", "wb"),
        )
        rc = await proc.wait()
        if rc != 0:
            print(f"Job s={s} p={p} failed with code {rc}")
        return rc

async def main():
    results = await asyncio.gather(*(run_exp(s, p) for s in range(480, 800) for p in [0.0005, 0.001, 0.002, 0.003]))
    if any(r != 0 for r in results):
        print("Some jobs failed.")
    else:
        print("All jobs succeeded.")

asyncio.run(main())