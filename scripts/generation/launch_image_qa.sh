#!/bin/bash
# Monitor volume QA sessions and launch image QA sessions when all complete

WORKDIR="${SGMRIQA_DATA_GENERATION:-data_generation}"
OUTDIR="$WORKDIR/qa_reasoning_data"
cd "$WORKDIR"

echo "$(date): Waiting for volume QA sessions to finish..."
echo "Monitoring: knee_train, knee_val, brain_train, brain_val"
echo "==========================================================="

while true; do
    running=0
    for session in knee_train knee_val brain_train brain_val; do
        if tmux has-session -t "$session" 2>/dev/null; then
            running=$((running + 1))
        fi
    done

    if [ "$running" -eq 0 ]; then
        echo ""
        echo "$(date): All volume sessions complete!"
        break
    fi

    echo "$(date): $running volume session(s) still running..."
    sleep 120  # Check every 2 minutes
done

echo ""
echo "==========================================================="
echo "$(date): Moving volume QA output files to qa_reasoning_data/..."
echo "==========================================================="

# Move any remaining volume QA files to the output folder
mv "$WORKDIR"/gpt4o_*_volume_qa*.json "$OUTDIR/" 2>/dev/null
echo "  Files moved:"
ls -lh "$OUTDIR"/*.json 2>/dev/null

echo ""
echo "==========================================================="
echo "$(date): Launching image QA sessions..."
echo "==========================================================="

# Knee Train Image QA (2375 diseased + 125 normal = 2500)
tmux new-session -d -s knee_train_img "cd $WORKDIR && python generate_qa_gpt4o_knee.py --input ../data_processing/knee/knee_train_volumes.json --data-root ../data --output qa_reasoning_data/gpt4o_knee_train_image_qa.json --max-samples 2500 --normal-sample-size 125 --image-only --checkpoint-every 50 --api-timeout 120"
echo "  Started: knee_train_img (2500 slices: 2375 diseased + 125 normal)"

# Knee Val Image QA (712 diseased + 38 normal = 750)
tmux new-session -d -s knee_val_img "cd $WORKDIR && python generate_qa_gpt4o_knee.py --input ../data_processing/knee/knee_val_volumes.json --data-root ../data --output qa_reasoning_data/gpt4o_knee_val_image_qa.json --max-samples 750 --normal-sample-size 38 --image-only --checkpoint-every 50 --api-timeout 120"
echo "  Started: knee_val_img (750 slices: 712 diseased + 38 normal)"

# Brain Train Image QA (2375 diseased + 125 normal = 2500)
tmux new-session -d -s brain_train_img "cd $WORKDIR && python generate_qa_gpt4o_brain.py --input ../data_processing/brain/brain_train_volumes.json --data-root ../data --output qa_reasoning_data/gpt4o_brain_train_image_qa.json --max-samples 2500 --normal-sample-size 125 --image-only --checkpoint-every 50 --api-timeout 120"
echo "  Started: brain_train_img (2500 slices: 2375 diseased + 125 normal)"

# Brain Val Image QA (712 diseased + 38 normal = 750)
tmux new-session -d -s brain_val_img "cd $WORKDIR && python generate_qa_gpt4o_brain.py --input ../data_processing/brain/brain_val_volumes.json --data-root ../data --output qa_reasoning_data/gpt4o_brain_val_image_qa.json --max-samples 750 --normal-sample-size 38 --image-only --checkpoint-every 50 --api-timeout 120"
echo "  Started: brain_val_img (750 slices: 712 diseased + 38 normal)"

echo ""
echo "==========================================================="
echo "$(date): All 4 image QA sessions launched!"
echo "  Output dir: $OUTDIR"
echo "  Total: 6,500 slices | 32,500 QA pairs | ~\$195 estimated"
echo "  Checkpoints saved every 50 samples"
echo "  API timeout: 120s per call"
echo "==========================================================="
