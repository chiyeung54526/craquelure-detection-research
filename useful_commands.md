## change python venv source
source /mnt/home/zou10/.venv/bin/activate
deactivate

## ask gpu/ cpu source
<!-- srun -p cpu -c 32 --mem=64G --time=01:00:00 --pty /bin/bash
srun -p rochester -w alphagpu14 --gres=gpu:1 --pty bash --gres=gpu:1 --time=04:30:00 --pty /bin/bash -->

# for CPU
salloc -p cpu --time=08:00:00 --job-name=cpu-env
salloc -p cpu --cpus-per-task=16 --time=08:00:00 --job-name=cpu-env
salloc -p cpu --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=cpu-env
# for GPU
salloc -p rochester --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=gpu-env

# then step into the assigned node
srun --pty bash

# check hostname 
hostname

# start jupyter notebook at a port
jupyter lab --no-browser --ip=0.0.0.0 --port=9854
<!-- nohup and & for no output in the shell -->
nohup jupyter notebook --no-browser --port=9854 &

## kernel management
jupyter kernelspec list

jupyter kernelspec remove kernel-name

<!-- tmux for creating new session window -->
tmux new -s jupyter_session

Ctrl + b  d for leaving to bash window

tmux attach -t jupyter_session
tmux new -s train
tmux attach -t train

# flush status every 10 seconds
watch -n 10 squeue -u $USER

python infer1221cnn.py  --ckpt weights/model_20260112_0414.ckpt-1 --test_dir myinput/testImages --threshold 0.51

# with random seed 3407
python infer1221cnn.py  --ckpt weights/best_model_20260218_0847.ckpt-1 --test_dir myinput/testImages --threshold 0.8
