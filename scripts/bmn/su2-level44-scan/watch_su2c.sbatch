#!/bin/bash
#SBATCH --job-name=fort-watch
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=log/watchc_%j.out
# T-027 phase-4 watcher for the per-column jobs (job name fort-su2c). col_jobs.txt lists "<jobid> <config> <heavy|light>".
# Every 10 min, for each column slot (list file + index) that left the queue without completing:
#   TIMEOUT / NODE_FAIL          -> resubmit with the same settings (while the slot has < 5 retries)
#   OUT_OF_MEMORY / hung worker  -> next config: heavy|light -> heavy4 (16 cores, 250 GB, 4 workers)
#                                   -> mid (24 cores, 380 GB, 2 workers) -> huge (32 cores, 500 GB, 2 workers)
#   anything else (incl. OOM in huge) -> NEEDS_ATTENTION, logged only
# mid sits before huge because a 380 GB job starts within hours on cs while a 500 GB one waits about two days.
cd ${SCRATCH}/fortuity-torch
touch col_jobs.txt col_retries.txt
LOG=log/watch_su2c.log
START=$(date +%s)
cfg_args() {  # $1 = config -> sbatch resource args
  case "$1" in
    heavy)  echo "--cpus-per-task=16 --mem=250G HWPAR_WORKERS=8" ;;
    light)  echo "--cpus-per-task=8 --mem=128G HWPAR_WORKERS=4" ;;
    heavy4) echo "--cpus-per-task=16 --mem=250G HWPAR_WORKERS=4" ;;
    huge)   echo "--cpus-per-task=32 --mem=500G HWPAR_WORKERS=2" ;;
    mid)    echo "--cpus-per-task=24 --mem=380G HWPAR_WORKERS=2" ;;
  esac
}
next_cfg() {  # memory escalation ladder
  case "$1" in heavy|light) echo heavy4 ;; heavy4) echo mid ;; mid) echo huge ;; esac
}
while true; do
  Q=$(squeue -h -r -u $USER -n fort-su2c -o %i)
  # latest record per (list, index): list derived from the job id via col_jobs.txt (job -> config -> list file)
  sacct -n -X -P -u $USER -S 2026-09-29T10:00 --name=fort-su2c -o JobID,State,End | grep -v "\[" \
    | awk -F'|' '{split($2,s," "); e=($3=="Unknown"||$3=="")?"9999":$3; print e, $1, s[1]}' | sort > latest_c.txt
  : > slots_c.txt
  while read e jid st; do
    base=${jid%_*}; idx=${jid#*_}; cfg=$(awk -v b=$base '$1==b{print $2}' col_jobs.txt); [ -z "$cfg" ] && continue
    list=cols_light.txt; [ "$cfg" != "light" ] && list=cols_heavy.txt; [ "$(awk -v b=$base '$1==b{print $3}' col_jobs.txt)" = "light" ] && list=cols_light.txt
    echo "$list:$idx $jid $st $cfg" >> slots_c.txt
  done < latest_c.txt
  awk '{last[$1]=$0} END {for (k in last) print last[k]}' slots_c.txt | sort > slots_latest.txt
  report=""; ndone=0
  while read slot jid st cfg; do
    [ "$st" = "COMPLETED" ] && { ndone=$((ndone+1)); continue; }
    echo "$Q" | grep -qx "$jid" && continue
    # a newer submission for the same slot may be queued: skip if any queued job maps to this slot
    queued=0; for q in $Q; do qb=${q%_*}; qi=${q#*_}; qc=$(awk -v b=$qb '$1==b{print $2}' col_jobs.txt); ql=cols_light.txt; [ "$qc" != "light" ] && ql=cols_heavy.txt; [ "$(awk -v b=$qb '$1==b{print $3}' col_jobs.txt)" = "light" ] && ql=cols_light.txt; [ "$ql:$qi" = "$slot" ] && queued=1; done
    [ $queued = 1 ] && continue
    n=$(grep -cx "$slot" col_retries.txt); list=${slot%%:*}; idx=${slot#*:}
    new=""
    if { [ "$st" = "TIMEOUT" ] || [ "$st" = "NODE_FAIL" ]; } && [ $n -lt 5 ]; then new=$cfg
    elif [ "$st" = "OUT_OF_MEMORY" ]; then new=$(next_cfg $cfg)
    fi
    if [ -n "$new" ]; then
      set -- $(cfg_args $new)
      j=$(sbatch --parsable --array=$idx $1 $2 --export=ALL,COLS=$list,$3 su2_col.sbatch 2>&1)
      tag=heavy; [ "$list" = "cols_light.txt" ] && tag=light
      echo "$j $new $tag" >> col_jobs.txt; echo "$slot" >> col_retries.txt
      echo "$(date '+%F %T') RESUBMITTED $slot ($st) as $new job $j" >> $LOG
    else report="$report $slot:$jid:$st"; fi
  done < slots_latest.txt
  # hang detection: a worker killed by the cgroup memory limit leaves the multiprocessing pool of the hw stage hung
  # (the job keeps RUNNING). Read each running job's oom_kill counter on its node; if > 0, cancel and resubmit.
  for id in $(squeue -h -r -u $USER -n fort-su2c -t RUNNING -o %i); do
    node=$(squeue -h -j $id -o %N); jid=$(squeue -h -j $id -o %A)
    k=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $node "grep -h '^oom_kill ' /sys/fs/cgroup/system.slice/slurmstepd.scope/job_${jid}/memory.events 2>/dev/null" 2>/dev/null | awk '{print $2}')
    [ -z "$k" ] && continue
    if [ "$k" -gt 0 ]; then
      base=${id%_*}; idx=${id#*_}; cfg=$(awk -v b=$base '$1==b{print $2}' col_jobs.txt); tag=$(awk -v b=$base '$1==b{print $3}' col_jobs.txt)
      list=cols_heavy.txt; { [ "$cfg" = "light" ] || [ "$tag" = "light" ]; } && list=cols_light.txt
      slot="$list:$idx"; n=$(grep -cx "$slot" col_retries.txt)
      scancel $id
      nxt=$(next_cfg $cfg)
      if [ -n "$nxt" ]; then
        set -- $(cfg_args $nxt)
        j=$(sbatch --parsable --array=$idx $1 $2 --export=ALL,COLS=$list,$3 su2_col.sbatch 2>&1)
        t2=heavy; [ "$list" = "cols_light.txt" ] && t2=light
        echo "$j $nxt $t2" >> col_jobs.txt; echo "$slot" >> col_retries.txt
        echo "$(date '+%F %T') HUNG worker killed (oom_kill=$k) in $id; cancelled and resubmitted $slot as $nxt job $j" >> $LOG
      else
        echo "$(date '+%F %T') HUNG worker killed (oom_kill=$k) in $id ($cfg); cancelled, NOT resubmitted: needs attention" >> $LOG
      fi
    fi
  done
  pend=$(squeue -h -r -u $USER -n fort-su2c -t PENDING | wc -l); run=$(squeue -h -r -u $USER -n fort-su2c -t RUNNING | wc -l)
  fails=$(cat results/su2_c_*.jsonl 2>/dev/null | grep -c '"check": false')
  echo "$(date '+%F %T') columns_done=$((423 + ndone))/446 pend=$pend run=$run checkfail=$fails NEEDS_ATTENTION:${report:- none}" >> $LOG
  if [ "$pend" = "0" ] && [ "$run" = "0" ]; then echo "$(date '+%F %T') DONE: no fort-su2c job left in the queue" >> $LOG; exit 0; fi
  if [ $(( $(date +%s) - START )) -gt $(( 47 * 3600 )) ]; then nj=$(sbatch --parsable watch_su2c.sbatch); echo "$(date '+%F %T') watcher handing over to job $nj" >> $LOG; exit 0; fi
  sleep 600
done
