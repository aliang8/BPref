for seed in 12345 12346 12347; do
    python train_SAC.py env=metaworld_button-press-v2 seed=$seed agent.actor_lr=0.0003 agent.critic_lr=0.0003  num_train_steps=1000000 agent.batch_size=512 double_q_critic.hidden_dim=256 double_q_critic.hidden_depth=3 diag_gaussian_actor.hidden_dim=256 diag_gaussian_actor.hidden_depth=3
done