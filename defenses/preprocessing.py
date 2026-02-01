"""
Preprocessing defenses for adversarial text attacks.

These defenses work at inference time by transforming input text before
classification. They can help counter character-level attacks like DeepWordBug.

Adapted from xarello/defenses/preprocessing.py for use with BODEGA.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import OpenAttack

# SEPARATOR used for text pairs in FC/C19 tasks
SEPARATOR = ' ~ '


class DefenseWrapper(OpenAttack.Classifier, ABC):
    """
    Base class for preprocessing defenses.

    Wraps an OpenAttack Classifier and intercepts inputs to apply
    defense transformations before passing to the victim model.
    """

    def __init__(self, victim: OpenAttack.Classifier, verbose: bool = False):
        self.victim = victim
        self.verbose = verbose
        self.modifications = []

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        defended_input = self.apply_defense(input_)
        return self.victim.get_prob(defended_input)

    def apply_defense(self, input_: List[str]) -> List[str]:
        result = []
        for text in input_:
            if SEPARATOR in text:
                # Handle text pairs (FC/C19 tasks)
                parts = text.split(SEPARATOR)
                defended_parts = [self.defend_single(p) for p in parts]
                defended_text = SEPARATOR.join(defended_parts)
            else:
                defended_text = self.defend_single(text)

            # Track and log modifications
            if text != defended_text:
                self.modifications.append((text, defended_text))
                if self.verbose:
                    print(f"\n[DEFENSE] Text modified:")
                    print(f"  Original: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"  Defended: {defended_text[:100]}{'...' if len(defended_text) > 100 else ''}")

            result.append(defended_text)
        return result

    def get_modifications(self) -> List[tuple]:
        return self.modifications

    def clear_modifications(self):
        self.modifications = []

    def save_modifications(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("original\tdefended\n")
            for orig, defended in self.modifications:
                # Escape tabs and newlines for TSV format
                orig_escaped = orig.replace('\t', '\\t').replace('\n', '\\n')
                defended_escaped = defended.replace('\t', '\\t').replace('\n', '\\n')
                f.write(f"{orig_escaped}\t{defended_escaped}\n")

    @abstractmethod
    def defend_single(self, text: str) -> str:
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()


class SpellCheckDefense(DefenseWrapper):
    """
    Defense that corrects spelling errors to counter character-level attacks.

    DeepWordBug and similar attacks use character swaps, insertions, and deletions
    which often result in misspellings. This defense attempts to correct them.
    """

    def __init__(self, victim: OpenAttack.Classifier, language: str = 'en', verbose: bool = False):
        super().__init__(victim, verbose)
        self.language = language
        self._spellchecker = None

    @property
    def spellchecker(self):
        if self._spellchecker is None:
            try:
                from symspellpy import SymSpell
                import importlib.resources
            except ImportError:
                raise ImportError(
                    "SpellCheckDefense requires symspellpy. "
                    "Install with: pip install symspellpy"
                )

            self._spellchecker = SymSpell(
                max_dictionary_edit_distance=2,
                prefix_length=7
            )

            # Load built-in English dictionary
            dictionary_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            if not self._spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1):
                raise RuntimeError("Failed to load symspellpy English dictionary")
        return self._spellchecker

    def defend_single(self, text: str) -> str:
        """Correct spelling errors in text."""
        words = text.split()
        corrected = []

        for word in words:
            # Preserve punctuation attached to words
            prefix = ''
            suffix = ''
            core = word

            # Strip leading punctuation
            while core and not core[0].isalnum():
                prefix += core[0]
                core = core[1:]

            # Strip trailing punctuation
            while core and not core[-1].isalnum():
                suffix = core[-1] + suffix
                core = core[:-1]

            if core:
                # Check if word needs correction
                from symspellpy import Verbosity
                suggestions = self.spellchecker.lookup(core.lower(), Verbosity.TOP, max_edit_distance=2)
                correction = suggestions[0].term if suggestions else None
                if correction and correction != core.lower():
                    # Preserve original case pattern
                    if core.isupper():
                        correction = correction.upper()
                    elif core[0].isupper():
                        correction = correction.capitalize()
                    corrected.append(prefix + correction + suffix)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)

        return ' '.join(corrected)


class CharacterNoiseDefense(DefenseWrapper):
    """
    Defense that adds character-level noise to input text using homoglyphs.

    By randomly perturbing characters, this defense can disrupt adversarial
    perturbations that rely on specific character patterns. Uses the homoglyphs
    library to find visually similar Unicode characters across different scripts
    (e.g., Cyrillic 'а' looks like Latin 'a').

    Requires: pip install homoglyphs
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(victim, verbose)
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self._homoglyphs = None
        # Cache for homoglyph lookups to avoid repeated calls
        self._cache = {}

    @property
    def homoglyphs(self):
        """Lazy initialization of homoglyphs library."""
        if self._homoglyphs is None:
            try:
                import homoglyphs as hg
            except ImportError:
                raise ImportError(
                    "CharacterNoiseDefense requires homoglyphs library. "
                    "Install with: pip install homoglyphs"
                )
            # Initialize with all character categories for maximum coverage
            self._homoglyphs = hg.Homoglyphs(categories=hg.Categories.get_all())
        return self._homoglyphs

    def _get_alternatives(self, char: str) -> List[str]:
        """Get visually similar alternatives for a character, with caching."""
        if char in self._cache:
            return self._cache[char]

        try:
            # Get all homoglyph combinations for this character
            variants = self.homoglyphs.get_combinations(char)
            # Filter out the original character
            alternatives = [v for v in variants if v != char and len(v) == 1]
            self._cache[char] = alternatives
            return alternatives
        except Exception:
            # If lookup fails, return empty list
            self._cache[char] = []
            return []

    def defend_single(self, text: str) -> str:
        """
        Apply character-level perturbation using homoglyphs.

        Randomly substitutes characters with visually similar Unicode alternatives
        based on the noise_std probability.
        """
        if self.noise_std == 0.0:
            return text

        result = []
        for char in text:
            # Skip spaces and control characters
            if char.isspace() or not char.isprintable():
                result.append(char)
                continue

            # Probability of perturbation scales with noise_std
            if self.rng.random() < self.noise_std:
                alternatives = self._get_alternatives(char)
                if alternatives:
                    replacement = self.rng.choice(alternatives)
                    result.append(replacement)
                else:
                    result.append(char)
            else:
                result.append(char)

        return ''.join(result)


class CharacterMaskingDefense(DefenseWrapper):
    """
    Defense that randomly masks (removes) characters from input.

    By randomly removing characters, this defense can disrupt adversarial
    perturbations that rely on specific character positions or sequences.

    Note: This is called "masking" rather than "dropout" to distinguish it
    from neural network dropout, which drops neurons during training.
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        masking_prob: float = 0.1,
        seed: Optional[int] = None,
        min_length: int = 10,
        verbose: bool = False
    ):
        super().__init__(victim, verbose)
        self.masking_prob = masking_prob
        self.min_length = min_length
        self.rng = random.Random(seed)

    def defend_single(self, text: str) -> str:
        """Randomly mask (remove) characters from text."""
        if self.masking_prob == 0.0:
            return text

        if len(text) <= self.min_length:
            return text

        # Calculate how many characters we can mask
        max_mask = len(text) - self.min_length

        # Decide which characters to keep
        kept = []
        masked_count = 0

        for char in text:
            # Always keep spaces to maintain word boundaries
            if char == ' ':
                kept.append(char)
            elif self.rng.random() >= self.masking_prob or masked_count >= max_mask:
                kept.append(char)
            else:
                masked_count += 1

        return ''.join(kept)


class IdentityDefense(DefenseWrapper):
    """No-op defense that passes input unchanged. Useful for baseline comparisons."""

    def __init__(self, victim: OpenAttack.Classifier, verbose: bool = False):
        super().__init__(victim, verbose)

    def defend_single(self, text: str) -> str:
        return text


def get_defense(
    defense_name: str,
    victim: OpenAttack.Classifier,
    param: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = False
) -> OpenAttack.Classifier:
    """
    Factory function to create defense wrappers.

    Args:
        defense_name: Type of defense ('none', 'spellcheck', 'char_noise', 'char_masking', 'identity')
        victim: The classifier to wrap
        param: Defense parameter (noise_std for char_noise, masking_prob for char_masking)
        seed: Random seed for reproducibility
        verbose: Whether to print modifications

    Returns:
        Wrapped classifier with defense applied, or original victim if defense_name is 'none'
    """
    defense_name = defense_name.lower()

    if defense_name == 'none' or defense_name == '':
        return victim
    elif defense_name == 'spellcheck':
        return SpellCheckDefense(victim, verbose=verbose)
    elif defense_name == 'char_noise':
        return CharacterNoiseDefense(victim, noise_std=param, seed=seed, verbose=verbose)
    elif defense_name == 'char_masking':
        return CharacterMaskingDefense(victim, masking_prob=param, seed=seed, verbose=verbose)
    elif defense_name == 'identity':
        return IdentityDefense(victim, verbose=verbose)
    else:
        raise ValueError(f"Unknown defense: {defense_name}. "
                        f"Available: none, spellcheck, char_noise, char_masking, identity")
