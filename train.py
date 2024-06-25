import torch
from tqdm import tqdm
import wandb
import utils

def train(policy, policy_old, train_dataloader, optimizer_policy, optimizer_value, device, config, scheduler_policy=None, scheduler_value=None):

    policy.train()
    policy_old.eval()
    assert policy_old.policy_net.training == False and policy_old.value_net.training == False, "Old policy should be in evaluation mode here"
    assert policy.policy_net.training == True and policy.value_net.training == True, "Policy should be in training mode here"
    for epoch in tqdm(range(config.epochs)):
        for batch, (states, actions, advantages, value_targets) in enumerate(train_dataloader):
            # normalize advantages between 0 and 1
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = states.to(device)
            actions = actions.to(device)
            advantages = advantages.to(device)
            value_targets = value_targets.to(device)
            
            dists, values = policy.actions_dist_and_v(states)
            old_dists = policy_old.actions_dist(states)

            log_probs = dists.log_prob(actions)
            old_log_probs = old_dists.log_prob(actions)

            # Equivalent of doing exp(log_probs) / exp(old_log_probs) 
            # but avoids overflows and division by (potentially if underflown) zero, breaking loss function
            ratios = torch.exp(log_probs - old_log_probs)

            # clipped surrogate loss
            l_clips = -torch.min(ratios * advantages, torch.clip(ratios, 1-config.eps_clip, 1+config.eps_clip) * advantages)
            loss_pi = torch.mean(l_clips)
            loss_entropy = dists.entropy().mean()
            loss_policy = loss_pi - config.entropy_bonus * loss_entropy

            # mse loss
            loss_value = torch.nn.functional.mse_loss(values, value_targets)

            # with two different optimizers
            loss_policy.backward()
            optimizer_policy.step()
            optimizer_policy.zero_grad()

            loss_value.backward()
            optimizer_value.step()
            optimizer_value.zero_grad()

            if utils.global_batch % config.log_frequency == 0:
                wandb.log({"train/loss_pi": loss_pi, 
                           "train/loss_v": loss_value,
                           "train/entropy": loss_entropy,
                           "train/lr_policy": optimizer_policy.param_groups[0]['lr'],
                           "train/lr_value": optimizer_value.param_groups[0]['lr'],
                           "train/batch": utils.global_batch})
            
            utils.global_batch += 1
        
        if scheduler_policy is not None:
            scheduler_policy.step()
        if scheduler_value is not None:
            scheduler_value.step()

        with torch.no_grad():
            # KL divergence between old and new policy for early stopping
            kl_div = torch.distributions.kl.kl_divergence(dists, old_dists).mean().item()
            wandb.log({"train/kl_div": kl_div, "train/batch": utils.global_batch})
            if kl_div > config.kl_limit:
                print(f"Early stopping at epoch {epoch} due to KL divergence {round(kl_div, 4)} > {config.kl_limit}")
                break