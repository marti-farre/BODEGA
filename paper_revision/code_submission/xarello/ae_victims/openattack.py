from tokenizers.decoders import WordPiece

from ae_victims.victim import Victim


class OpenAttackVictimWrapper(Victim):
    def __init__(self, oa_victim, tokeniser):
        self.oa_victim = oa_victim
        self.tokeniser = tokeniser
        self.decoder = WordPiece()
    
    def pred_changed(self, tokens_original, tokens_modified, tokens_previous):
        text_original = self.recover_text(tokens_original)
        text_modified = self.recover_text(tokens_modified)
        text_previous = self.recover_text(tokens_previous)
        probs = self.oa_victim.get_prob([text_original, text_modified, text_previous])
        if probs[0].argmax() != probs[1].argmax():
            # Decision change, collect reward
            result = 1.0
        elif probs[0][0] > probs[0][1]:
            # Original decision was 0, gain is increased probability of 1
            result = probs[1][1] - probs[2][1]
        else:
            # Original decision was 1, gain is increased probability of 0
            result = probs[1][0] - probs[2][0]
        return result
    
    def recover_text(self, tokens):
        result = self.decoder.decode(self.tokeniser.convert_ids_to_tokens(tokens))
        result = result.replace('[CLS]', ' ').replace('[SEP]', ' ').replace('[PAD]', ' ').strip()
        return result
    
    def pred(self, tokens_original):
        text_original = self.recover_text(tokens_original)
        result = self.oa_victim.get_prob([text_original]).argmax(axis=1)[0]
        return result
