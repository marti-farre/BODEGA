import OpenAttack
import numpy as np
import torch
from OpenAttack.tags import Tag
from tokenizers.decoders import WordPiece

from agent.q_learning import Qmodel
from utilities.llm_utils import prepare_for_llm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class QAttacker(OpenAttack.attackers.ClassificationAttacker):
    TAGS = {Tag("english", "lang"), Tag("get_pred", "victim")}
    
    def __init__(self, model_path, llm_name, env, local_device, device, long_text):
        if model_path:
            print("Loading model...")
            self.q_model = Qmodel(llm_name, env.BEST_K, env.candidate_embedding_size)
            self.q_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict = False)
            params_all = count_parameters(self.q_model)
            params_BERT = count_parameters(self.q_model.llm)
            print("loaded with "+str(params_all)+" paramaters, "+(str(params_all-params_BERT))+" own.")
        else:
            self.q_model = None
        self.env = env
        self.local_device = local_device
        self.device = device
        if model_path:
            self.q_model.to(device)
        self.previous_actions = None
        self.decoder = WordPiece()
        self.long_text = long_text
    
    def attack(self, victim, input_, goal):
        orig_preds = victim.get_pred([input_])
        self.env.current_text += 1
        assert (self.env.orig_texts[self.env.current_text] == input_)
        for MAX_STEPS in [5, 10, 25, 50]:
            self.previous_actions = []
            max_attempts = int(50 / MAX_STEPS)
            for j in range(max_attempts):
                current_tokens = self.env.orig_tokens[self.env.current_text].copy()
                for i in range(MAX_STEPS * (5 if self.long_text else 1)):
                    input_ids, attention_mask, token_type_ids = prepare_for_llm([current_tokens], self.env.tokeniser)
                    if self.q_model:
                        candidate_representations = torch.tensor(
                            np.array([self.env.replacement_embeddings[self.env.current_text]])).type('torch.FloatTensor')
                        with torch.no_grad():
                            result_all = self.q_model(input_ids.to(self.device), attention_mask.to(self.device),
                                                      token_type_ids.to(self.device), candidate_representations.to(self.device))
                        result_perpred = torch.stack([result_all[i, orig_preds[i], :, :] for i in range(len(orig_preds))])
                        result_perpred = result_perpred.to(self.local_device).numpy()
                    else:
                        result_perpred = np.random.rand(1, self.env.MAX_TEXT_LENGTH, self.env.BEST_K)
                    computed_Qs = self.hard_rules(result_perpred, attention_mask)[0]
                    which_acceptable = (computed_Qs != -1.0)
                    if np.sum(which_acceptable) != 0:
                        computed_Qs[~which_acceptable] = -np.inf
                    best = np.unravel_index([np.argmax(computed_Qs)], computed_Qs.shape)
                    action_location = best[0][0]
                    action_replacement = best[1][0]
                    action = (action_location, action_replacement)
                    replacement_token = self.env.replacement_tokens[self.env.current_text][action_location][
                        action_replacement]
                    current_tokens[action_location] = replacement_token
                    self.previous_actions.append(action)
                    x_new = self.decoder.decode(self.env.tokeniser.convert_ids_to_tokens(current_tokens)).replace(
                        '[PAD]', '').replace('[SEP]', '').replace('[CLS]', '').replace('[UNK]', '').strip() + \
                            self.env.overflow[self.env.current_text]
                    y_new = victim.get_pred([x_new])
                    if goal.check(x_new, y_new):
                        return x_new
        return None
    
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
