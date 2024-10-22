#!/usr/bin/env python3

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Any

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path, help="Path to .gr file containing a PCFG")
    parser.add_argument("sentences", type=Path, help="Path to .sen file containing tokenized input sentences")
    parser.add_argument("-s", "--start_symbol", type=str, default="ROOT", help="Start symbol of the grammar")
    parser.add_argument("--progress", action="store_true", default=False, help="Display a progress bar")

    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING)

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        self.tokens = tokens
        self.grammar = grammar.specialize(tokens)  # Use specialized grammar for efficiency
        self.progress = progress
        self.profile: CounterType[str] = Counter()
        self.predicted_nonterminals = [set() for _ in range(len(tokens) + 1)]
        self.cols = [Agenda() for _ in range(len(tokens) + 1)]
        self._run_earley()

    def accepted(self) -> bool:
        """Check if the sentence is accepted by the grammar."""
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol and 
                item.next_symbol() is None and 
                item.start_position == 0):
                return True
        return False

    def _run_earley(self) -> None:
        """Fill the chart using Earley's algorithm."""
        self._predict(self.grammar.start_symbol, 0)

        for i, column in tqdm.tqdm(enumerate(self.cols), total=len(self.cols), disable=not self.progress):
            while column:
                item = column.pop()
                next_symbol = item.next_symbol()
                if next_symbol is None:
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next_symbol):
                    self._predict(next_symbol, i)
                else:
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Predict rules for the given nonterminal at the specified position."""
        if nonterminal in self.predicted_nonterminals[position]:
            return  # Avoid redundant predictions
        self.predicted_nonterminals[position].add(nonterminal)

        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position, weight=rule.weight)
            self.cols[position].push(new_item)
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Advance the dot if the next token matches."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            scanned_token = self.tokens[position]
            new_item = item.with_dot_advanced(child=scanned_token)
            self.cols[position + 1].push(new_item)
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach a completed item to waiting items."""
        mid = item.start_position
        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced(weight_increment=item.weight, child=item)
                self.cols[position].push(new_item)
                self.profile["ATTACH"] += 1


class Agenda:
    """A collection of items to be processed, with duplicate detection."""

    def __init__(self) -> None:
        self._items: List[Item] = []
        self._index: Dict[Item, int] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add an item to the agenda unless it is already present with a lower weight."""
        idx = self._index.get(item)
        if idx is None:
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        elif item.weight < self._items[idx].weight:
            self._items[idx] = item
            if idx < self._next:
                self._next = idx

    def pop(self) -> Item:
        """Remove and return the next item."""
        if not self:
            raise IndexError("Agenda is empty")
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Return all items."""
        return self._items


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        with open(file) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                prob, lhs, rhs = line.split("\t")
                rule = Rule(lhs, tuple(rhs.split()), -math.log2(float(prob)))
                self._expansions.setdefault(lhs, []).append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self._expansions

    def specialize(self, tokens: List[str]) -> Grammar:
        """Create a specialized grammar for the given tokens."""
        specialized_grammar = Grammar(self.start_symbol)
        token_set = set(tokens)
        for lhs, rules in self._expansions.items():
            specialized_rules = [
                rule for rule in rules if all(
                    sym in token_set or self.is_nonterminal(sym) for sym in rule.rhs
                )
            ]
            if specialized_rules:
                specialized_grammar._expansions[lhs] = specialized_rules
        return specialized_grammar


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    weight: float

    def __repr__(self) -> str:
        return f"{self.lhs} → {' '.join(self.rhs)}"


@dataclass(frozen=True)
class Item:
    rule: Rule
    dot_position: int
    start_position: int
    weight: float
    children: Tuple[Any, ...] = ()

    def next_symbol(self) -> Optional[str]:
        if self.dot_position == len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self, weight_increment: float = 0.0, child=None) -> Item:
        return Item(
            rule=self.rule,
            dot_position=self.dot_position + 1,
            start_position=self.start_position,
            weight=self.weight + weight_increment,
            children=self.children + (child,)
        )


def print_parse(item: Item) -> str:
    if not item.children:
        return f"({item.rule.lhs})"
    return f"({item.rule.lhs} {' '.join(print_parse(child) if isinstance(child, Item) else str(child) for child in item.children)})"


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f:
            sentence = sentence.strip()  # Remove leading/trailing whitespace

            if not sentence:
                continue
            tokens = sentence.strip().split()
            if tokens:
                chart = EarleyChart(tokens, grammar, args.progress)
                best_item = min((item for item in chart.cols[-1].all() if item.rule.lhs == args.start_symbol and item.next_symbol() is None and item.start_position == 0), key=lambda x: x.weight, default=None)
                if best_item:
                    print(f"{print_parse(best_item)} {best_item.weight}")
                else:
                    print("NONE")
            else:
                print("NONE")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    main()
