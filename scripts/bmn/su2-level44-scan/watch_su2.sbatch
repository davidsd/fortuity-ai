#!/bin/bash
#SBATCH --job-name=fort-watch
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=log/watch_%j.out
# T-027 on-cluster watcher (runs on Torch itself, so it does not depend on the laptop's ssh connection).
# Every 10 min: tasks of the SU(2) scan (job name fort-su2) that left the queue without completing are resubmitted
#   TIMEOUT                         -> ENGINE=par, 8 cores, 128 GB, 48 h   (at most 3 times per task)
#   OUT_OF_MEMORY on a serial job   -> same                                (at most 3 times per task)
#   anything else (FAILED, CANCELLED, OOM of an ENGINE=par job) -> only logged under NEEDS_ATTENTION
# One status line per cycle goes to log/watch_su2.log. Exits when no fort-su2 task is queued; resubmits itself
# near its own wall-time limit if the scan is still running.
cd ${SCRATCH}/fortuity-torch
touch par_jobs.txt retries.txt
LOG=log/watch_su2.log
START=$(date +%s)
while true; do
  RUNQ=$(squeue -h -r -u $USER -n fort-su2 -o %i | sed -E "s/^[0-9]+_//" | sort -n)
  # -P: pipe-delimited, so states with spaces ("CANCELLED by 1234") cannot shift the columns
  sacct -n -X -P -u $USER -S 2026-09-28T21:40 --name=fort-su2 -o JobID,State,End | grep -v "\[" \
    | awk -F'|' '{split($2,s," "); e=($3=="Unknown"||$3=="")?"9999":$3; print e, $1, s[1]}' | sort \
    | awk '{split($2,a,"_"); last[a[2]]=a[1]" "$3} END {for (t in last) print t, last[t]}' | sort -n > latest_w.txt
  retry=""; report=""
  while read t job st; do
    echo "$RUNQ" | grep -qx "$t" && continue
    [ "$st" = "COMPLETED" ] && continue
    ispar=$(grep -cx "$job" par_jobs.txt); n=$(grep -cx "$t" retries.txt)
    if [ "$st" = "TIMEOUT" ] && [ "$n" -lt 3 ]; then retry="$retry,$t"
    elif [ "$st" = "OUT_OF_MEMORY" ] && [ "$ispar" = "0" ] && [ "$n" -lt 3 ]; then retry="$retry,$t"
    else report="$report ${job}_$t:$st"; fi
  done < latest_w.txt
  retry=${retry#,}
  if [ -n "$retry" ]; then
    j=$(sbatch --parsable --array=$retry --cpus-per-task=8 --mem=128G --time=48:00:00 --export=ALL,ENGINE=par su2_scan.sbatch 2>&1)
    echo "$j" >> par_jobs.txt; echo "$retry" | tr "," "\n" >> retries.txt
    echo "$(date '+%F %T') RESUBMITTED tasks $retry on ENGINE=par (128G, 48h) as job $j" >> $LOG
  fi
  done_t=$(awk '$3=="COMPLETED"{print $1}' latest_w.txt | grep -vxF -f <(echo "$RUNQ"; echo x) | wc -l)
  pend=$(squeue -h -r -u $USER -n fort-su2 -t PENDING | wc -l); run=$(squeue -h -r -u $USER -n fort-su2 -t RUNNING | wc -l)
  secs=$(cat results/su2_t*.jsonl 2>/dev/null | wc -l); fails=$(cat results/su2_t*.jsonl 2>/dev/null | grep -c '"check": false')
  echo "$(date '+%F %T') tasks_done=$done_t/64 pend=$pend run=$run sectors=$secs checkfail=$fails NEEDS_ATTENTION:${report:- none}" >> $LOG
  if [ "$pend" = "0" ] && [ "$run" = "0" ]; then echo "$(date '+%F %T') DONE: no fort-su2 task left in the queue" >> $LOG; exit 0; fi
  if [ $(( $(date +%s) - START )) -gt $(( 47 * 3600 )) ]; then
    nj=$(sbatch --parsable watch_su2.sbatch); echo "$(date '+%F %T') watcher handing over to job $nj" >> $LOG; exit 0
  fi
  sleep 600
done
