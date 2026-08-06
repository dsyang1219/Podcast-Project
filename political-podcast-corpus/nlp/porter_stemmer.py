"""Classic Porter (1980) stemming algorithm, self-contained (no nltk/regex
dependency -- this environment's nltk install trips a sandbox import guard
on its `regex` dependency). Used ONLY for the optional `stem_comparison`
regime (nlp/preprocess_regimes.py), which exists to demonstrate stemming is
worse than lemmatization per Schofield & Mimno (2016) / Denny & Spirling
(2018), not as a candidate default.

Public-domain algorithm; this is a standard, widely-used pure-Python port
of Porter's original ANSI C reference implementation.
"""
from __future__ import annotations


class PorterStemmer:
    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i):
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return i == self.k0 or not self._cons(i - 1)
        return True

    def _m(self):
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        return any(not self._cons(i) for i in range(self.k0, self.j + 1))

    def _doublec(self, j):
        if j < self.k0 + 1:
            return False
        return self.b[j] == self.b[j - 1] and self._cons(j)

    def _cvc(self, i):
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s):
        length = len(s)
        if s[-1] != self.b[self.k]:
            return False
        if length > (self.k - self.k0 + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        self.b = self.b[:self.j + 1] + s
        self.k = self.j + len(s)

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                if self.b[self.k - 1] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + "i"

    def _step2(self):
        table = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
            ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
            ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
        ]
        for suf, rep in table:
            if self._ends(suf):
                self._r(rep)
                break

    def _step3(self):
        table = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
                  ("ical", "ic"), ("ful", ""), ("ness", "")]
        for suf, rep in table:
            if self._ends(suf):
                self._r(rep)
                break

    def _step4(self):
        table = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
                 "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
        for suf in table:
            if self._ends(suf):
                if self._m() > 1:
                    self.k = self.j
                break
        else:
            if self._ends("ion") and self.j >= self.k0 and self.b[self.j] in "st":
                if self._m() > 1:
                    self.k = self.j

    def _step5(self):
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        word = word.lower()
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        if self.k <= self.k0 + 1:
            return word
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[self.k0:self.k + 1]
