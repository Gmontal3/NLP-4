#!/usr/bin/env python3

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple
from typing import Any

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing tokens with grammar.  
        progress says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        # Batch duplicate check: Track predicted nonterminals per column
        self.predicted_nonterminals = [set() for _ in range(len(self.tokens) + 1)]

        # Vocabulary specialization: Use a specialized grammar
        self.grammar = self.grammar.specialize(self.tokens)

        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return True
        return False   # we didn't find any appropriate item

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next_symbol = item.next_symbol()
                if next_symbol is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next_symbol):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next_symbol, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        # Batch duplicate check: Skip if already predicted in this column
        if nonterminal in self.predicted_nonterminals[position]:
            return  # Already predicted this nonterminal at this position
        self.predicted_nonterminals[position].add(nonterminal)

        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self.cols[position].push(new_item)
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            scanned_token = self.tokens[position]
            new_item = item.with_dot_advanced(child=scanned_token)
            self.cols[position + 1].push(new_item)
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        mid = item.start_position
        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced(weight_increment=item.weight, child=item)
                self.cols[position].push(new_item)
                self.profile["ATTACH"] += 1

class Agenda:
    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._next = 0                     # index of first item that has not yet been popped

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables len(my_agenda)."""
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added with a lower weight."""
        idx = self._index.get(item)
        if idx is None:
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        else:
            existing_item = self._items[idx]
            if item.weight < existing_item.weight:
                self._items[idx] = item
                if idx < self._next:
                    self._next = idx
            else:
                pass

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string representation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        self.terminals: set = set()
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line)."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                prob, lhs, rhs = line.split("\t")
                prob = float(prob)
                rhs = tuple(rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight= -math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)
                # Collect terminals
                for symbol in rhs:
                    if not self.is_nonterminal(symbol):
                        self.terminals.add(symbol)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions

    def specialize(self, tokens: List[str]) -> Grammar:
        """Return a new Grammar instance that only includes rules that produce terminals in tokens."""
        specialized_grammar = Grammar(self.start_symbol)
        terminals_in_sentence = set(tokens)
        for lhs, rules in self._expansions.items():
            specialized_rules = []
            for rule in rules:
                # Check if rule's terminals are in the sentence
                terminals_in_rule = [symbol for symbol in rule.rhs if not self.is_nonterminal(symbol)]
                if all(t in terminals_in_sentence for t in terminals_in_rule):
                    specialized_rules.append(rule)
            if specialized_rules:
                specialized_grammar._expansions[lhs] = specialized_rules
        return specialized_grammar

@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"

@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart."""
    rule: Rule
    dot_position: int
    start_position: int
    weight: float = 0.0
    backpointer: Optional[Tuple[Any, Any]] = None
    children: Tuple[Any, ...] = ()
    
    def next_symbol(self) -> Optional[str]:
        """Next unprocessed symbol."""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self, weight_increment: float = 0.0, child=None) -> Item:
        new_dot_position = self.dot_position + 1
        new_weight = self.weight + weight_increment
        if new_dot_position == len(self.rule.rhs):
            new_weight += self.rule.weight
        new_children = self.children + (child,)
        return Item(
            rule=self.rule,
            dot_position=new_dot_position,
            start_position=self.start_position,
            weight=new_weight,
            children=new_children,
        )

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule}, weight={self.weight:.4f})"

def print_parse(item: Item) -> str:
    if len(item.children) == 0:
        if len(item.rule.rhs) == 0:
            return f"({item.rule.lhs})"
        else:
            return f"({item.rule.lhs} {' '.join(item.rule.rhs)})"
    else:
        child_strings = []
        for child in item.children:
            if isinstance(child, Item):
                child_strings.append(print_parse(child))
            else:
                child_strings.append(child)
        return f"({item.rule.lhs} {' '.join(child_strings)})"

def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence:
                tokens = sentence.split()
                chart = EarleyChart(tokens, grammar, progress=args.progress)
                best_item = None
                for item in chart.cols[-1].all():
                    if (item.rule.lhs == args.start_symbol and item.next_symbol() is None and item.start_position == 0):
                        if best_item is None or item.weight < best_item.weight:
                            best_item = item
                if best_item:
                    print(f"{print_parse(best_item)} (Weight: {best_item.weight:.14f})")
                else:
                    print("NONE")
            else:
                print("NONE")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()