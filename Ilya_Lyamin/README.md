# RL-RND

## RND на MiniGrid

Baseline: `use_rnd=false` — чистый PPO из `train.py`.
RND: `use_rnd=true, rnd_beta=0.1` — `IntrinsicRewardWrapper` добавляет `β·r_int` к награде.

Запуск (из корня репозитория):
  python train.py --map MiniGrid-MultiRoom-N2-S4-v0 --run-dir laptop_cpu_ver0 \
    --fixed-params '{"max_episode_steps":200,"total_timesteps":50000,
    "no_improve_episodes":1000,"n_success":5,"success_window":100,
    "reward_delta":0.01,"use_rnd":true,"rnd_beta":0.1}'

Логи: `<run_dir>/<map>/reward.txt`, `steps_per_ep.txt`, `fps.txt`, `intrinsic_reward.txt`
(эпизод, Σ r_int_scaled, средний rnd_loss, beta), чекпоинт `rnd_module.pt`.

Упрощения относительно Burda et al. (2018): одна value-голова в PPO (без раздельных
ext/int value и GAE), r_int добавляется к r_ext внутри wrapper'а (SB3 учится как обычно),
без клиппинга r_int и без раздельного учёта эпизодов по типам.
