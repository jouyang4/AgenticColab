# Data transfer

`colab upload` / `colab download` move **one file at a time** through the runtime proxy.
Fine for scripts and modest artifacts; unreliable for big ones.

## 1. Chunk large uploads

A ~161 MB `colab upload` dies with `SSLEOFError: EOF occurred in violation of protocol`.
Split → upload → reassemble → verify by hash avoids it.

```bash
split -b 16M bigfile.tgz chunk_
for c in chunk_*; do
  colab upload -s "$SESS" "$c" "/content/chunks/$c"   # with an external timeout
done
```

```python
import hashlib, subprocess
subprocess.run("cat /content/chunks/chunk_* > /content/bigfile.tgz", shell=True, check=True)
print("MD5", hashlib.md5(open("/content/bigfile.tgz","rb").read()).hexdigest())
```

- Compare against local `md5sum`. `scripts/upload-big.sh` implements the whole loop
  (split → upload with retry → `cat` → md5 compare → cleanup).
- 16 MB is a good default chunk. Smaller = more round trips; larger reintroduces the failure.
- **Never wrap an upload in `set -e` inside a bootstrap.** A failed big upload then aborts
  the deploy before the extraction step, leaving a session holding tarballs and no scripts,
  which fails later as a confusing `No such file or directory`.

## 2. Separate timeout budget for transfers

```bash
cc()    { local t=$1; shift; timeout "$t" colab --auth=oauth2 "$@"; }
ccbig() { cc 1200 "$@"; }

cc    90 status -s "$SESS"
ccbig    upload -s "$SESS" ./ckpt.tgz /content/ckpt.tgz
```

A 300 s cap cannot move a 500 MB checkpoint, and the failure is silent: the restore step
finds no checkpoint and the job restarts from zero. Watch for a checkpoint that **grows**
over a campaign and outgrows even the big budget — cap it or prune old state.

## 3. Snapshot atomically

```python
import subprocess
subprocess.run(["bash","-lc",
  "cd /content && tar czf state_$(date +%s).tgz run_state/ && echo TAR_OK"], check=True)
```

Downloading a live, actively-written file gives a torn copy. Worst for database-like files
(HDF5, SQLite) where a torn copy may still *open* and report wrong contents.

## 4. Bank cumulatively

```bash
mkdir -p bank/
colab download -s "$SESS" /content/out.tgz "incoming_$(date +%s).tgz"
tar xzf "incoming_$(date +%s).tgz" -C bank/      # merge
tar czf all_results.tgz -C bank .                # rebuild from the merged bank
```

Over a campaign, results come from a succession of VMs. Overwriting one `results.tgz` per VM
silently destroys the previous machine's output.

## 5. Always attempt the download

The CLI reports failure for cells that completed server-side (short HTTP read timeouts on
the control path). Make the download unconditional.

```bash
cc 90 exec -s "$SESS" -f ./final_step.py || echo "exec reported failure; downloading anyway"
ccbig download -s "$SESS" /content/result.json ./result.json || echo "download failed"
```

## 6. Sync the stream, not just the final artifact

If a job produces a stream (trajectory chunks, per-epoch checkpoints, per-item results),
sync **every poll**. Banking only the end-state summary is how a long run across several VM
lifetimes yields a file covering only the last fraction of the work. One file per completed
unit + stage scripts that skip existing outputs makes "sync" a copy of whatever is new and
restart free.

## 7. Many small idempotent units, not one long terminal step

Cut work into units of a few minutes; after each, write its own output file and download it.

- a dropped connection costs one unit, not the run;
- restart re-runs the chain in seconds because completed stages skip;
- you never hold the only copy of anything important on a machine you don't control.

On ≤0.6.0 add the hard constraint that anything written after ~50 min of session age may be
unreachable, so **download before the long terminal step**. On ≥0.7.x the handle refreshes
itself and that deadline is gone; the unit discipline still is what makes long runs survivable.

## 8. Google Drive

`colab drivemount [path] -s NAME` mounts at `/content/drive` by default. Persists past the VM.

- interactive auth may be required depending on credential path;
- not POSIX semantics — don't use it as a high-IOPS scratch target. Write to `/content`, sync
  in batches;
- **deletes through a mounted Drive propagate to Drive.** With a sync tool watching the
  folder, a delete on the VM is a delete everywhere. Treat it as shared mutable state; prefer
  writing new paths over mutating existing ones.

For machine-to-machine banking, object storage or your own host is simpler and faster.

## 9. Disk on the VM

Per-shape size and sequential write throughput: [01](01-machine-shapes.md).

- `/content` is sized generously (hundreds of GB on larger shapes) but dies with the VM. It
  is scratch, not storage.
- Sequential write is the number that matters for checkpointing. A read measured right after
  a write is page cache — you cannot drop caches without root, so treat every such read
  figure as cache-warm.
