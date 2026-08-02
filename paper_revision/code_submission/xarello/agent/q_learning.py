import random
from typing import List, Any

import numpy as np
import torch
from torch.nn import Module, Linear, ReLU
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer

from agent.basic_agents import Agent
from env.EnvAE import copy_observation
from utilities.llm_utils import prepare_for_llm


class Qlearner(Agent):
    MAX_EPS = 1.0
    MIN_EPS = 0.10
    DISCOUNTING = 0.5
    MEMORY_SIZE = 4000
    MINIBATCH_SIZE = 16
    
    def __init__(self, env, llm_name, warmup_episodes, device=torch.device('cpu'), outfile=None, longterm_memory=False, starting_episodes = 0):
        self.replay_memory: List[Any] = [None] * self.MEMORY_SIZE
        self.memory_index = 0
        
        self.q_model = Qmodel(llm_name, env.BEST_K, env.candidate_embedding_size)
        # Disable dropout
        self.q_model.eval()
        self.local_device = torch.device('cpu')
        self.device = device
        self.q_model.to(device)
        learning_rate = 2e-05  # 1e-3
        self.optimiser = Adam(self.q_model.parameters(), lr=learning_rate)
        self.tokeniser = AutoTokenizer.from_pretrained(llm_name)
        
        self.eps = self.MAX_EPS
        self.replacement_embeddings = env.replacement_embeddings
        
        self.previous_actions = []
        self.episodes = starting_episodes
        self.current_text = 0
        self.warmup_episodes = warmup_episodes
        self.only_greedy = False
        self.longterm_memory = longterm_memory
        
        self.outfile = outfile
    
    def use_env(self, env):
        self.replacement_embeddings = env.replacement_embeddings
    
    def choose_action(self, observation):
        if observation[0] != self.current_text:
            self.previous_actions = []
            self.current_text = observation[0]
        if random.random() < self.eps and (not self.only_greedy):
            action = self.random_action(observation)
        else:
            action = self.greedy_action(observation)
        self.previous_actions.append(action)
        return action
    
    def learn(self, observation_old, action, observation_new, reward, terminated, truncated):
        # Store the experience in memory
        experience = (copy_observation(observation_old), action, copy_observation(observation_new), reward,
                      terminated)
        self.replay_memory[self.memory_index] = experience
        self.memory_index = (self.memory_index + 1) % self.MEMORY_SIZE
        if self.memory_index % self.MINIBATCH_SIZE == 0:
            # Recall experiences from memory
            if self.longterm_memory:
                experiences = random.choices(self.valid_experiences(), k=self.MINIBATCH_SIZE)
            else:
                experiences = self.valid_experiences()
            # Learn from them
            self.learn_from_experiences(experiences)
            if not self.longterm_memory:
                self.replay_memory: List[Any] = [None] * self.MEMORY_SIZE
                self.memory_index=0
        # Update the exploration factor
        self.eps = self.MIN_EPS if self.episodes >= self.warmup_episodes else (self.MAX_EPS - (
                (self.MAX_EPS - self.MIN_EPS) * (self.episodes * 1.0 / self.warmup_episodes)))
        # Keep track of the episodes to update epsilon
        if terminated or truncated:
            self.episodes = self.episodes + 1
    
    def learn_from_experiences(self, experiences):
        observations_old = [x[0] for x in experiences]
        action_locations = np.array([x[1][0] for x in experiences])
        action_replacements = np.array([x[1][1] for x in experiences])
        observations_new = [x[2] for x in experiences]
        rewards = np.array([x[3] for x in experiences])
        terminateds = np.array([x[4] for x in experiences])
        if self.DISCOUNTING > 0.0:
            next_Q = self.compute_Qs(observations_new, with_gradient=False)
            future_reward = (1 - terminateds) * self.DISCOUNTING * np.max(next_Q, axis=(1, 2))
        else:
            future_reward = 0.0
        observed_Q = torch.tensor(rewards + future_reward).type('torch.FloatTensor').to(self.device)
        expected_Q = self.compute_Qs(observations_old, with_gradient=True)
        expected_Q = expected_Q[np.arange(expected_Q.shape[0]), action_locations, action_replacements]
        loss_value = torch.sum((expected_Q - observed_Q) ** 2)
        # loss_detached = loss_value.detach().cpu().numpy()
        # print("LOSS: " + str(loss_detached))
        # if self.outfile:
        #    self.outfile.write(str(loss_detached) + '\n')
        self.take_loss(loss_value)
    
    def compute_Qs(self, observations, with_gradient):
        current_is = [o[0] for o in observations]
        orig_preds = [o[1] for o in observations]
        input_ids, attention_mask, token_type_ids = prepare_for_llm([o[2] for o in observations], self.tokeniser)
        candidate_representations = torch.tensor(
            np.array([self.replacement_embeddings[current_i] for current_i in current_is])).type(
            'torch.FloatTensor')
        if with_gradient:
            result_all = self.q_model(input_ids.to(self.device), attention_mask.to(self.device),
                                      token_type_ids.to(self.device), candidate_representations.to(self.device))
        else:
            with torch.no_grad():
                result_all = self.q_model(input_ids.to(self.device), attention_mask.to(self.device),
                                          token_type_ids.to(self.device), candidate_representations.to(self.device))
        result_perpred = torch.stack([result_all[i, orig_preds[i], :, :] for i in range(len(orig_preds))])
        if with_gradient:
            return result_perpred
        else:
            result_perpred = result_perpred.to(self.local_device).numpy()
            result_perpred = self.hard_rules(result_perpred, attention_mask)
            return result_perpred
    
    def greedy_action(self, observation):
        computed_Qs = self.compute_Qs([observation], with_gradient=False)[0]
        which_acceptable = (computed_Qs != -1.0)
        if np.sum(which_acceptable) != 0:
            computed_Qs[~which_acceptable] = -np.inf
        best = np.unravel_index([np.argmax(computed_Qs)], computed_Qs.shape)
        action_location = best[0][0]
        action_replacement = best[1][0]
        action = (action_location, action_replacement)
        print("GREEDY " + str(action) + ": " + str(computed_Qs[action_location, action_replacement]))
        return action
    
    def random_action(self, observation):
        computed_Qs = self.compute_Qs([observation], with_gradient=False)[0]
        which_acceptable = (computed_Qs != -1.0).flatten()
        if np.sum(which_acceptable) != 0:
            best = random.choices(range(len(which_acceptable)), which_acceptable * 1, k=1)[0]
        else:
            best = random.choices(range(len(which_acceptable)), k=1)[0]
        best = np.unravel_index([best], computed_Qs.shape)
        action_location = best[0][0]
        action_replacement = best[1][0]
        action = (action_location, action_replacement)
        print("RANDOM " + str(action) + ": " + str(computed_Qs[action[0], action[1]]))
        return action
    
    def take_loss(self, loss_value):
        self.optimiser.zero_grad()
        loss_value.backward()
        self.optimiser.step()
    
    def valid_experiences(self):
        if self.replay_memory[-1] is not None:
            return self.replay_memory
        else:
            return self.replay_memory[:self.memory_index]
    
    def hard_rules(self, q_values, attention_mask):
        # Can't modify the CLS token
        q_values[:, 0, :] = -1.0
        # Can't modify the words where there aren't any
        q_values[attention_mask == 0, :] = -1.0
        # Can't replace a token with the same token
        for previous_action in self.previous_actions:
            action_location, action_replacement = previous_action
            q_values[:, action_location, action_replacement] -= 0.1
        return q_values
    
    def save_to_path(self,path):
        torch.save(self.q_model.state_dict(), path)
    
    def load_from_path(self,path):
        self.q_model.load_state_dict(torch.load(path))

class Qmodel(Module):
    def __init__(self, llm_name, candidates_number, candidate_size):
        super(Qmodel, self).__init__()
        self.candidates_number = candidates_number
        self.candidate_size = candidate_size
        self.llm = AutoModel.from_pretrained(llm_name)
        # linear input is LLM output plus candidate representation
        linear_input = self.llm.config.hidden_size + candidate_size
        # separate output for each confusion direction
        linear_output = 2
        self.first_linear = Linear(linear_input, 8)
        self.relu = ReLU()
        self.second_linear = Linear(8, linear_output)
        # self.freeze_llm()
    
    def forward(self, input_ids, attention_mask, token_type_ids, candidate_representation):
        original_representation = self.llm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[
            'last_hidden_state']  # shape 32 x 128 x 768
        original_representation_repeated = original_representation.repeat(1, 1, self.candidates_number).reshape(
            original_representation.shape[0], original_representation.shape[1], self.candidates_number,
            original_representation.shape[2])  # shape 32 x 128 x 10 x 768
        full_representation = torch.cat((original_representation_repeated, candidate_representation),
                                        dim=-1)  # shape 32 x 128 x 10 x (768+768)
        after_linear = self.first_linear(full_representation)
        contextualised = self.relu(after_linear)  # shape 32 x 128 x 10 x 8
        reduced = self.second_linear(contextualised)  # shape 32 x 128 x 10 x 2
        reordered = torch.movedim(reduced, 3, 1)  # shape 32 x 2 x 128 x 10
        return reordered
    
    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True
