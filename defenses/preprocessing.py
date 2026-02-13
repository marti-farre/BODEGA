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


class UnicodeCanonicalizationDefense(DefenseWrapper):
    """
    Defense that normalizes Unicode text to remove adversarial artifacts.

    Character-level adversarial attacks often exploit Unicode features:
    - Homoglyphs: visually similar chars from different scripts (Cyrillic 'а' vs Latin 'a')
    - Zero-width characters: invisible chars like ZWSP (U+200B)
    - Confusables: characters that look similar but have different code points

    This defense counters these by:
    1. Removing zero-width and invisible characters
    2. Applying NFKC Unicode normalization
    3. Mapping common confusables to ASCII equivalents

    Based on: Bhalerao et al. "Data-driven mitigation of adversarial text perturbation"
    Survey reference: Section 5.1.2 - Perturbation Correction
    """

    # Zero-width and invisible characters to remove
    ZERO_WIDTH_CHARS = {
        '\u200b',  # Zero Width Space (ZWSP)
        '\u200c',  # Zero Width Non-Joiner (ZWNJ)
        '\u200d',  # Zero Width Joiner (ZWJ)
        '\u200e',  # Left-to-Right Mark
        '\u200f',  # Right-to-Left Mark
        '\u2060',  # Word Joiner
        '\u2061',  # Function Application
        '\u2062',  # Invisible Times
        '\u2063',  # Invisible Separator
        '\u2064',  # Invisible Plus
        '\ufeff',  # Byte Order Mark / Zero Width No-Break Space
        '\u00ad',  # Soft Hyphen
        '\u034f',  # Combining Grapheme Joiner
        '\u061c',  # Arabic Letter Mark
        '\u115f',  # Hangul Choseong Filler
        '\u1160',  # Hangul Jungseong Filler
        '\u17b4',  # Khmer Vowel Inherent Aq
        '\u17b5',  # Khmer Vowel Inherent Aa
        '\u180e',  # Mongolian Vowel Separator
        '\u2800',  # Braille Pattern Blank
        '\u3164',  # Hangul Filler
        '\uffa0',  # Halfwidth Hangul Filler
    }

    # Common confusable mappings (homoglyphs to ASCII)
    # Based on Unicode confusables.txt and common attack patterns
    CONFUSABLES_MAP = {
        # Cyrillic lowercase confusables
        '\u0430': 'a',  # Cyrillic small a
        '\u0435': 'e',  # Cyrillic small ie
        '\u0456': 'i',  # Cyrillic small byelorussian-ukrainian i
        '\u043e': 'o',  # Cyrillic small o
        '\u0440': 'p',  # Cyrillic small er
        '\u0441': 'c',  # Cyrillic small es
        '\u0443': 'y',  # Cyrillic small u
        '\u0445': 'x',  # Cyrillic small ha
        '\u0455': 's',  # Cyrillic small dze
        '\u0458': 'j',  # Cyrillic small je
        # Cyrillic uppercase confusables
        '\u0410': 'A',  # Cyrillic capital A
        '\u0412': 'B',  # Cyrillic capital Ve
        '\u0415': 'E',  # Cyrillic capital Ie
        '\u041a': 'K',  # Cyrillic capital Ka
        '\u041c': 'M',  # Cyrillic capital Em
        '\u041d': 'H',  # Cyrillic capital En
        '\u041e': 'O',  # Cyrillic capital O
        '\u0420': 'P',  # Cyrillic capital Er
        '\u0421': 'C',  # Cyrillic capital Es
        '\u0422': 'T',  # Cyrillic capital Te
        '\u0425': 'X',  # Cyrillic capital Ha
        # Greek confusables
        '\u03b1': 'a',  # Greek small alpha
        '\u03b5': 'e',  # Greek small epsilon
        '\u03b9': 'i',  # Greek small iota
        '\u03bf': 'o',  # Greek small omicron
        '\u03c1': 'p',  # Greek small rho
        '\u03c5': 'u',  # Greek small upsilon
        '\u0391': 'A',  # Greek capital Alpha
        '\u0392': 'B',  # Greek capital Beta
        '\u0395': 'E',  # Greek capital Epsilon
        '\u0397': 'H',  # Greek capital Eta
        '\u0399': 'I',  # Greek capital Iota
        '\u039a': 'K',  # Greek capital Kappa
        '\u039c': 'M',  # Greek capital Mu
        '\u039d': 'N',  # Greek capital Nu
        '\u039f': 'O',  # Greek capital Omicron
        '\u03a1': 'P',  # Greek capital Rho
        '\u03a4': 'T',  # Greek capital Tau
        '\u03a7': 'X',  # Greek capital Chi
        '\u03a5': 'Y',  # Greek capital Upsilon
        '\u0396': 'Z',  # Greek capital Zeta
        # Various typographic symbols
        '\u2010': '-',  # Hyphen
        '\u2011': '-',  # Non-breaking hyphen
        '\u2012': '-',  # Figure dash
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2015': '-',  # Horizontal bar
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201a': "'",  # Single low-9 quotation mark
        '\u201b': "'",  # Single high-reversed-9 quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u201e': '"',  # Double low-9 quotation mark
        '\u201f': '"',  # Double high-reversed-9 quotation mark
        '\u2032': "'",  # Prime
        '\u2033': '"',  # Double prime
        '\u2039': '<',  # Single left-pointing angle quotation mark
        '\u203a': '>',  # Single right-pointing angle quotation mark
        '\u00ab': '"',  # Left-pointing double angle quotation mark
        '\u00bb': '"',  # Right-pointing double angle quotation mark
        # Fullwidth characters to ASCII
        '\uff01': '!',  # Fullwidth exclamation mark
        '\uff1f': '?',  # Fullwidth question mark
        '\uff0e': '.',  # Fullwidth full stop
        '\uff0c': ',',  # Fullwidth comma
        # Mathematical and other symbols
        '\u2212': '-',  # Minus sign
        '\u00d7': 'x',  # Multiplication sign
        '\u2217': '*',  # Asterisk operator
        '\u2219': '.',  # Bullet operator
        '\u00b7': '.',  # Middle dot
    }

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        normalize_form: str = 'NFKC',
        remove_zero_width: bool = True,
        map_confusables: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Unicode canonicalization defense.

        Args:
            victim: The classifier to wrap
            normalize_form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
                           NFKC is recommended as it handles compatibility characters
            remove_zero_width: Whether to remove zero-width invisible characters
            map_confusables: Whether to map confusable characters to ASCII
            verbose: Whether to print modifications
        """
        super().__init__(victim, verbose)
        self.normalize_form = normalize_form
        self.remove_zero_width = remove_zero_width
        self.map_confusables = map_confusables

    def defend_single(self, text: str) -> str:
        """Apply Unicode canonicalization to text."""
        import unicodedata

        result = text

        # Step 1: Remove zero-width and invisible characters
        if self.remove_zero_width:
            result = ''.join(c for c in result if c not in self.ZERO_WIDTH_CHARS)

        # Step 2: Apply Unicode normalization
        # NFKC: Compatibility decomposition + canonical composition
        # This handles ligatures, fullwidth chars, etc.
        result = unicodedata.normalize(self.normalize_form, result)

        # Step 3: Map confusable characters to ASCII equivalents
        if self.map_confusables:
            result = ''.join(self.CONFUSABLES_MAP.get(c, c) for c in result)

        return result


class MajorityVoteDefense(DefenseWrapper):
    """
    Defense that creates perturbed copies of input and uses majority voting.

    For each input text:
    1. Create N perturbed copies using random character-level perturbations
    2. Classify each copy with the victim model
    3. Return the majority vote prediction (as probability distribution)

    This defense exploits the fact that adversarial perturbations are often
    fragile to additional small random changes. By voting across multiple
    perturbed versions, we can "vote out" the adversarial effect.

    IMPORTANT: This defense overrides get_prob() directly (not just defend_single)
    because it needs to query the victim model multiple times and aggregate results.

    Based on: Swenor & Kalita "Using random perturbations to mitigate adversarial attacks"
    Survey reference: Section 5.1.1 - Perturbation Identification
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        num_copies: int = 5,
        perturbation_prob: float = 0.1,
        aggregation: str = 'hard',
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize majority vote defense.

        Args:
            victim: The classifier to wrap
            num_copies: Number of perturbed copies to create (odd numbers preferred for voting)
            perturbation_prob: Probability of perturbing each character
            aggregation: Voting method - 'hard' (count predictions) or 'soft' (average probs)
            seed: Random seed for reproducibility
            verbose: Whether to print modifications
        """
        super().__init__(victim, verbose)
        self.num_copies = num_copies
        self.perturbation_prob = perturbation_prob
        self.aggregation = aggregation
        self.rng = random.Random(seed)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """
        Override get_prob to implement majority voting.

        Instead of transforming input once, we:
        1. Generate num_copies perturbed versions of each input
        2. Get predictions for all versions
        3. Aggregate via majority vote (hard or soft)
        """
        all_probs = []

        for i in range(self.num_copies):
            # Generate perturbed copy (using parent's apply_defense for text pair handling)
            perturbed_input = self.apply_defense(input_)
            # Get probabilities from victim
            probs = self.victim.get_prob(perturbed_input)
            all_probs.append(probs)

        # Stack: shape (num_copies, batch_size, num_classes)
        all_probs = np.stack(all_probs, axis=0)

        if self.aggregation == 'hard':
            # Hard voting: count predictions and convert to probability distribution
            predictions = all_probs.argmax(axis=2)  # (num_copies, batch_size)
            batch_size = predictions.shape[1]
            num_classes = all_probs.shape[2]
            result = np.zeros((batch_size, num_classes))

            for sample_idx in range(batch_size):
                votes = predictions[:, sample_idx]
                for class_idx in range(num_classes):
                    result[sample_idx, class_idx] = np.sum(votes == class_idx) / self.num_copies

            return result
        else:  # soft voting
            # Soft voting: average probabilities across all copies
            return np.mean(all_probs, axis=0)

    def defend_single(self, text: str) -> str:
        """
        Apply random perturbation to a single text.

        Perturbation types:
        - Delete: Remove a character
        - Swap: Swap adjacent characters
        - Insert: Insert a random character
        """
        if self.perturbation_prob == 0.0:
            return text

        result = []
        i = 0
        while i < len(text):
            char = text[i]

            # Always preserve spaces to maintain word boundaries
            if char.isspace():
                result.append(char)
                i += 1
                continue

            if self.rng.random() < self.perturbation_prob:
                # Choose a random perturbation
                perturbation = self.rng.choice(['delete', 'swap', 'insert'])

                if perturbation == 'delete':
                    # Skip this character (delete it)
                    i += 1
                    continue
                elif perturbation == 'swap' and i < len(text) - 1 and not text[i + 1].isspace():
                    # Swap with next character
                    result.append(text[i + 1])
                    result.append(char)
                    i += 2
                    continue
                elif perturbation == 'insert':
                    # Insert a random lowercase letter before current char
                    random_char = chr(self.rng.randint(ord('a'), ord('z')))
                    result.append(random_char)
                    result.append(char)
                    i += 1
                    continue

            # Default: keep the character as-is
            result.append(char)
            i += 1

        return ''.join(result)


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
        defense_name: Type of defense ('none', 'spellcheck', 'char_noise', 'char_masking',
                      'identity', 'unicode', 'majority_vote')
        victim: The classifier to wrap
        param: Defense parameter (noise_std for char_noise, masking_prob for char_masking,
               num_copies for majority_vote)
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
    elif defense_name == 'unicode' or defense_name == 'unicode_canonicalization':
        return UnicodeCanonicalizationDefense(victim, verbose=verbose)
    elif defense_name == 'majority_vote' or defense_name == 'vote':
        num_copies = int(param) if param > 0 else 5
        return MajorityVoteDefense(victim, num_copies=num_copies, seed=seed, verbose=verbose)
    else:
        raise ValueError(f"Unknown defense: {defense_name}. "
                        f"Available: none, spellcheck, char_noise, char_masking, identity, "
                        f"unicode, majority_vote")
