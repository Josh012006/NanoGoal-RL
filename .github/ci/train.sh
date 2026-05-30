#!/bin/bash
# .github/ci/train.sh
# Runs in background on the droplet, detached from the GitHub Actions job.
# Usage: bash .github/ci/train.sh <flag_file> <sha> <branch> <work_dir>

FLAG_FILE="$1"
SHA="$2"
BRANCH="$3"
WORK_DIR="$4"
VENV="$WORK_DIR/.venv/bin"

cd "$WORK_DIR"

# ── Load SendGrid config ───────────────────────────────────────────────────────
if [ -f ~/.sendgrid_config ]; then
  source ~/.sendgrid_config
fi
NOTIFY_EMAIL="josuesmjr.mongan@gmail.com"
FROM_EMAIL="josuesmjr.mongan@gmail.com"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

send_email() {
  local subject="$1"
  local body="$2"
  if [ -z "$SENDGRID_API_KEY" ]; then
    log "SENDGRID_API_KEY not set — skipping email."
    return
  fi
  local escaped_body
  escaped_body=$(echo "$body" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
  curl -s -o /dev/null -X POST https://api.sendgrid.com/v3/mail/send \
    -H "Authorization: Bearer $SENDGRID_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"personalizations\":[{\"to\":[{\"email\":\"$NOTIFY_EMAIL\"}]}],\"from\":{\"email\":\"$FROM_EMAIL\"},\"subject\":\"$subject\",\"content\":[{\"type\":\"text/plain\",\"value\":\"$escaped_body\"}]}" || true
}

# ── Read flags ────────────────────────────────────────────────────────────────
TRAIN_EASY=false
TRAIN_MEDIUM=false
TRAIN_HARD=false

while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  case "$key" in
    train_easy)   TRAIN_EASY="$value"   ;;
    train_medium) TRAIN_MEDIUM="$value" ;;
    train_hard)   TRAIN_HARD="$value"   ;;
  esac
done < "$FLAG_FILE"

TRAINING_FAILED=false
log "Flags → easy=$TRAIN_EASY medium=$TRAIN_MEDIUM hard=$TRAIN_HARD"
send_email "🚀 NanoGoal training started ($SHA)" \
  "Training started on branch $BRANCH.\n\nFlags:\n- easy=$TRAIN_EASY\n- medium=$TRAIN_MEDIUM\n- hard=$TRAIN_HARD\n\nCommit: $SHA"

# ── CPU watcher ───────────────────────────────────────────────────────────────
# Alerts if average CPU usage stays below 20% for more than 10 consecutive checks (5 min)
cpu_watcher() {
  local low_count=0
  local threshold=20
  local max_low=10  # 10 checks × 30s = 5 minutes
  log "[CPU watcher] Started."
  while true; do
    sleep 30
    local usage
    usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
    usage=${usage%.*}
    if [ "${usage:-0}" -lt "$threshold" ] 2>/dev/null; then
      low_count=$((low_count + 1))
      log "[CPU watcher] Low CPU usage: ${usage}% (${low_count}/${max_low})"
      if [ "$low_count" -ge "$max_low" ]; then
        log "[CPU watcher] ⚠️ CPU usage has been low for 5 minutes — training may have crashed."
        send_email "⚠️ NanoGoal training — low CPU detected" \
          "CPU usage has been below ${threshold}% for 5 consecutive minutes.\n\nThis may indicate the training has crashed or stalled.\n\nCommit: $SHA\nBranch: $BRANCH\n\nCheck the droplet: tail -f $WORK_DIR/logs/train_session.log"
        low_count=0
      fi
    else
      low_count=0
    fi
  done
}

# ── 2h log reporter ───────────────────────────────────────────────────────────
log_reporter() {
  local interval=$((2 * 3600))
  while true; do
    sleep $interval
    local body="Training progress report — $(date -u '+%Y-%m-%d %H:%M UTC')\nCommit: $SHA\nBranch: $BRANCH\n\n"
    for difficulty in easy medium hard; do
      local logfile="$WORK_DIR/logs/train_${difficulty}.log"
      if [ -f "$logfile" ]; then
        body+="=== $difficulty ===\n"
        body+="$(tail -30 "$logfile")\n\n"
      fi
    done
    send_email "📊 NanoGoal training report — $(date -u '+%H:%M UTC')" "$body"
    log "[Log reporter] 2h report sent."
  done
}

# Launch both background processes
cpu_watcher &
CPU_WATCHER_PID=$!
log "CPU watcher started (PID $CPU_WATCHER_PID)"

log_reporter &
LOG_REPORTER_PID=$!
log "Log reporter started (PID $LOG_REPORTER_PID)"

# Cleanup function — kills background processes when train.sh exits
cleanup() {
  log "Stopping background processes..."
  kill "$CPU_WATCHER_PID" 2>/dev/null || true
  kill "$LOG_REPORTER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Train easy ────────────────────────────────────────────────────────────────
if [ "$TRAIN_EASY" = "true" ]; then
  log "Starting easy training..."
  send_email "🟢 NanoGoal — easy training started" "Easy training started.\nCommit: $SHA"

  if $VENV/python -u train_easy.py > logs/train_easy.log 2>&1; then
    log "Easy training complete."
    log "Running easy evaluations..."
    $VENV/python eval.py 0 0 >> logs/train_easy.log 2>&1
    $VENV/python eval.py 0 1 >> logs/train_easy.log 2>&1
    $VENV/python eval.py 0 2 >> logs/train_easy.log 2>&1
    $VENV/python saving_plots.py results/easy/ppo_eval_easy.csv   plots/easy 0   >> logs/train_easy.log 2>&1
    $VENV/python saving_plots.py results/easy/ppo_eval_medium.csv plots/easy 1 0 >> logs/train_easy.log 2>&1
    $VENV/python saving_plots.py results/easy/ppo_eval_hard.csv   plots/easy 2 0 >> logs/train_easy.log 2>&1
    touch logs/train_easy.DONE
    send_email "✅ NanoGoal — easy training complete" "Easy training finished successfully.\nCommit: $SHA"
    log "Easy eval and plots done."
  else
    touch logs/train_easy.FAILED
    send_email "❌ NanoGoal — easy training FAILED" \
      "Easy training failed.\nCommit: $SHA\n\nLast logs:\n$(tail -50 logs/train_easy.log)"
    log "Easy training FAILED."
    TRAINING_FAILED=true
  fi
fi

# ── Train medium ──────────────────────────────────────────────────────────────
if [ "$TRAIN_MEDIUM" = "true" ] && [ "${TRAINING_FAILED:-false}" = "false" ]; then
  log "Starting medium training..."
  send_email "🟡 NanoGoal — medium training started" "Medium training started.\nCommit: $SHA"

  if $VENV/python -u train_medium.py > logs/train_medium.log 2>&1; then
    log "Medium training complete."
    log "Running medium evaluations..."
    $VENV/python eval.py 1 0 >> logs/train_medium.log 2>&1
    $VENV/python eval.py 1 1 >> logs/train_medium.log 2>&1
    $VENV/python eval.py 1 2 >> logs/train_medium.log 2>&1
    $VENV/python saving_plots.py results/medium/ppo_eval_easy.csv   plots/medium 0 0 >> logs/train_medium.log 2>&1
    $VENV/python saving_plots.py results/medium/ppo_eval_medium.csv plots/medium 1   >> logs/train_medium.log 2>&1
    $VENV/python saving_plots.py results/medium/ppo_eval_hard.csv   plots/medium 2 0 >> logs/train_medium.log 2>&1
    touch logs/train_medium.DONE
    send_email "✅ NanoGoal — medium training complete" "Medium training finished successfully.\nCommit: $SHA"
    log "Medium eval and plots done."
  else
    touch logs/train_medium.FAILED
    send_email "❌ NanoGoal — medium training FAILED" \
      "Medium training failed.\nCommit: $SHA\n\nLast logs:\n$(tail -50 logs/train_medium.log)"
    log "Medium training FAILED."
    TRAINING_FAILED=true
  fi
fi

# ── Train hard ────────────────────────────────────────────────────────────────
if [ "$TRAIN_HARD" = "true" ] && [ "${TRAINING_FAILED:-false}" = "false" ]; then
  log "Starting hard training..."
  send_email "🔴 NanoGoal — hard training started" "Hard training started.\nCommit: $SHA"

  if $VENV/python -u train_hard.py > logs/train_hard.log 2>&1; then
    log "Hard training complete."
    log "Running hard evaluations..."
    $VENV/python eval.py 2 0 >> logs/train_hard.log 2>&1
    $VENV/python eval.py 2 1 >> logs/train_hard.log 2>&1
    $VENV/python eval.py 2 2 >> logs/train_hard.log 2>&1
    $VENV/python saving_plots.py results/hard/ppo_eval_easy.csv   plots/hard 0 0 >> logs/train_hard.log 2>&1
    $VENV/python saving_plots.py results/hard/ppo_eval_medium.csv plots/hard 1 0 >> logs/train_hard.log 2>&1
    $VENV/python saving_plots.py results/hard/ppo_eval_hard.csv   plots/hard 2   >> logs/train_hard.log 2>&1
    touch logs/train_hard.DONE
    send_email "✅ NanoGoal — hard training complete" "Hard training finished successfully.\nCommit: $SHA"
    log "Hard eval and plots done."
  else
    touch logs/train_hard.FAILED
    send_email "❌ NanoGoal — hard training FAILED" \
      "Hard training failed.\nCommit: $SHA\n\nLast logs:\n$(tail -50 logs/train_hard.log)"
    log "Hard training FAILED."
    TRAINING_FAILED=true
  fi
fi

# ── Reset train.flag ──────────────────────────────────────────────────────────
log "Resetting train.flag..."
{
  echo "train=false"
  echo "train_easy=false"
  echo "train_medium=false"
  echo "train_hard=false"
  echo "generate_cache=false"
} > "$FLAG_FILE"

# ── Commit & push ─────────────────────────────────────────────────────────────
log "Committing results..."

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git add .
git reset topology_cache.dir topology_cache.bak topology_cache.dat topology_cache.db 2>/dev/null || true

if [ -z "$(git status --porcelain)" ]; then
  log "No changes to commit."
else
  git commit -m "feat: Trained models after commit of ID: $SHA [skip ci]"
  git push \
    https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git \
    "HEAD:${BRANCH}"
  log "Pushed to branch $BRANCH."
fi

# ── Create GitHub issue ───────────────────────────────────────────────────────
log "Creating GitHub issue..."

BODY="Models trained from commit: $SHA"$'\n\n'
[ "$TRAIN_EASY"   = "true" ] && { [ -f logs/train_easy.DONE   ] && BODY+="✅ Easy training completed"$'\n'   || BODY+="❌ Easy training failed"$'\n';   }
[ "$TRAIN_MEDIUM" = "true" ] && { [ -f logs/train_medium.DONE ] && BODY+="✅ Medium training completed"$'\n' || BODY+="❌ Medium training failed"$'\n'; }
[ "$TRAIN_HARD"   = "true" ] && { [ -f logs/train_hard.DONE   ] && BODY+="✅ Hard training completed"$'\n'   || BODY+="❌ Hard training failed"$'\n';   }
BODY+=$'\n''- [ ] Generate and review training metrics'$'\n'
BODY+='- [ ] Validate model performance'$'\n'
BODY+='- [ ] Make video demos of the models performance'$'\n'
BODY+='- [ ] Update README'

gh issue create \
  --title "Training finished for $SHA" \
  --body "$BODY"

# ── Final notification ────────────────────────────────────────────────────────
send_email "🏁 NanoGoal — all training complete" \
  "All training sessions have completed.\n\n$BODY\n\nCommit: $SHA\nBranch: $BRANCH"

log "All done."