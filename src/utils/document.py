import re
from typing import List


def word_tokenize(sentence: str) -> List[str]:
        """Splits a sentence into word list using regex.

        Args:
            sentence:
                Input sentence

        Returns:
            List of words.
        """
        pat = re.compile(r"[\w']+|[.,!?;]")
        if isinstance(sentence, str):
            return pat.findall(sentence.lower())
        else:
            return []