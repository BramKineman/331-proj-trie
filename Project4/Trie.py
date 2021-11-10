"""
Your Name
Project 4 - Tries
CSE 331 Fall 2020
Professor Sebnem Onsay
"""

from __future__ import annotations
from typing import Tuple, Dict, List


class TrieNode:
    """
    Implementation of a trie node.
    """

    # DO NOT MODIFY

    __slots__ = "children", "is_end"

    def __init__(self, arr_size: int = 26) -> None:
        """
        Constructs a TrieNode with arr_size slots for child nodes.
        :param arr_size: Number of slots to allocate for child nodes.
        :return: None
        """
        self.children = [None] * arr_size
        self.is_end = 0

    def __str__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        if self.empty():
            return "..."
        children = self.children  # to shorten proceeding line
        return str(
            {chr(i + ord("a")) + "*" * min(children[i].is_end, 1): children[i] for i in range(26) if children[i]})

    def __repr__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        return self.__str__()

    def __eq__(self, other: TrieNode) -> bool:
        """
        Compares two TrieNodes for equality.
        :return: True if two TrieNodes are equal, else False
        """
        if not other or self.is_end != other.is_end:
            return False
        return self.children == other.children

    # Implement Below

    def empty(self) -> bool:
        """
        Determine if the TrieNode is a leaf, meaning it has no children.
        :return: True if TrieNode is empty, False otherwise.
        """
        for x in self.children:
            if x is not None:
                return False
        return True

    @staticmethod
    def _get_index(char: str) -> int:
        """
        Retrieves the index of a character as it corresponds to the alphabet.
        Lower and uppercase letters are treated the same.
        :param char: character to retrieve the index of.
        :return: Integer of the alphabetical index the character corresponds to.
        """
        return ord(char.lower()) - 97

    def get_child(self, char: str) -> TrieNode:
        """
        Retrieves TrieNode with passed character.
        :param char: character of child TrieNode to retrieve.
        :return: TriedNode
        """
        return self.children[self._get_index(char)]

    def set_child(self, char: str) -> None:
        """
        Creates a TrieNode and stores it in children at the correct index.
        :param char: character of child TrieNode to create
        :return: None.
        """
        index = self._get_index(char)
        self.children[index] = TrieNode()

    def delete_child(self, char: str) -> None:
        """
        Deletes the child TrieNode at the correct index.
        :param char: character of child TrieNode to delete.
        :return: None.
        """
        index = self._get_index(char)
        self.children[index] = None


class Trie:
    """
    Implementation of a trie.
    """

    # DO NOT MODIFY

    __slots__ = "root", "unique", "size"

    def __init__(self) -> None:
        """
        Constructs an empty Trie.
        :return: None.
        """
        self.root = TrieNode()
        self.unique = 0
        self.size = 0

    def __str__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return "Trie Visual:\n" + str(self.root)

    def __repr__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return self.__str__()

    def __eq__(self, other: Trie) -> bool:
        """
        Compares two Tries for equality.
        :return: True if two Tries are equal, else False
        """
        return self.root == other.root

    # Implement Below

    def add(self, word: str) -> int:
        """
        Adds passed word to Trie, incrementing unique if the word does not exist in Trie,
        incrementing is_end for the last character of the string, and incrementing size.
        :param word: String to be added to the Trie.
        :return: Number of times word exists in the trie.
        """

        # end of word  ->  index == len(word)
        # we want to go through each character in the word to add
        # O(k) , k is length of string
        def add_inner(node: TrieNode, index: int) -> int:
            """
            Helper method for add function.
            :param node: Root node of sub-trie to add word into.
            :param index: Integer index of the current character being traversed/added in word.
            :return: Number of times word exists in the trie.
            """
            if len(word) - 1 == index:  # reached the end of the word
                if node.get_child(word[index]) is None:
                    node.set_child(word[index])
                if node.get_child(word[index]).is_end == 0:
                    self.unique += 1

                node.get_child(word[index]).is_end += 1
                return node.get_child(word[index]).is_end

            if node.get_child(word[index]) is None:  # check to see if node is already created at desired index
                node.set_child(word[index])  # if it doesn't exist, set that index with that char.

            return add_inner(node.get_child(word[index]), index + 1)  # recursive call with new node and new index

        self.size += 1
        return add_inner(self.root, 0)

    def search(self, word: str) -> int:
        """
        Traverses the trie and finds how many occurrences there are of the searched for string.
        :param word: Word to search for.
        :return: Number of occurrences of the word.
        """

        def search_inner(node: TrieNode, index: int) -> int:
            """
            Traverses the trie starting at the root node. Called recursively.
            :param node: Root node of sub-trie to search word in.
            :param index: Integer index of the current character in searched word.
            """
            if len(word) - 1 == index and node.get_child(word[index]):  # base case: if we found the whole word
                return node.get_child(word[index]).is_end
            if node.get_child(word[index]) is None:  # if character is not there/word does not exist
                return 0
            return search_inner(node.get_child(word[index]), index + 1)

        return search_inner(self.root, 0)

    def delete(self, word: str) -> int:
        """
        Removes word from Trie. Updates unique and size variables.
        :param word: Word to be deleted from Trie.
        :return: Integer number of times word existed before deletion, 0 if word is not found in the Trie.
        """

        def delete_inner(node: TrieNode, index: int) -> Tuple[int, bool]:
            """
            Helper method to remove a word from Trie. Updates unique and size variables.
            :param node: Root TrieNode to traverse, locate, and delete word from.
            :param index: Integer index of the current character in word to delete.
            :return: Tuple at each node indicating the number of copies of word deleted and
            if it should be pruned.
            """
            if len(word) == index:  # base case, I am at the end of the word I want to delete
                # set is_end, unique, size
                is_end = node.is_end
                if node.is_end > 0:
                    self.unique -= 1
                    self.size -= node.is_end
                node.is_end = 0
                return is_end, node.empty()

            # does the child exist?
            if node.get_child(word[index]):
                num_words, needs_pruned = delete_inner(node.get_child(word[index]), index + 1)
                # check if we can delete child
                if needs_pruned:
                    node.delete_child(word[index])
                return num_words, node.is_end == 0 and node.empty()

            return 0, False

        return delete_inner(self.root, 0)[0]

    def __len__(self) -> int:
        """
        Retrieves the total number of words in the vocabulary.
        :return: Integer number of words in the vocabulary.
        """
        return self.size

    def __contains__(self, word: str) -> bool:
        """
        Determines if passed word is within the Trie.
        :param word: Word to determine existence.
        :return: True if word is in trie, false otherwise.
        """
        return self.search(word) > 0

    def empty(self) -> bool:
        """
        Determines if the vocabulary of the trie is empty.
        :return: True if vocabulary is empty, false otherwise.
        """
        return self.__len__() == 0

    def get_vocabulary(self, prefix: str = "") -> Dict[str, int]:
        """
        Retrieves a dictionary of all words starting with the passed prefix.
        :param prefix: Prefix of words requested for retrieval.
        :return: Dictionary object with key word and value amount of that word.
        """
        vocab = {}
        prefix_node = self.root
        if prefix != "":
            for c in prefix:
                if prefix_node.get_child(c) is not None:
                    prefix_node = prefix_node.get_child(c)
                else:
                    return {}

        def get_vocabulary_inner(node, suffix):
            """
            Helper function that traverses beginning at the suffix all words
            that began with prefix.
            :param node: Root node of subtrie to add nodes from.
            :param suffix: The string of letters which must be appended to prefix to arrive at current node.
            """
            # base case
            if node.is_end > 0:
                vocab[suffix] = node.is_end
            if not node.empty():
                for num in range(0, 26):
                    if node.children[num] is not None:
                        get_vocabulary_inner(node.children[num], suffix + chr(num + 97))
            return vocab

        return get_vocabulary_inner(prefix_node, prefix)

    def autocomplete(self, word: str) -> Dict[str, int]:
        """
        Creates a dictionary consisting of words in Trie that match the template of word.
        :param word: Template string to match with words in Trie.
        :return: Dictionary of {word, count} pairs containing every word in the Trie which matches the template
        of word, where periods in word may be filled with any character.
        """
        if self.root.empty():
            return {}
        # hint, declare a dictionary in the outer scope and add items to it in inner
        matching = {}

        def autocomplete_inner(node, prefix, index) -> None:
            """
            Helper method to traverse subtrie.
            :param node: Root node of subtrie to traverse.
            :param prefix: String that is recursively concatinated on to.
            :param index: Integer holding index used to access word characters.
            """
            # Base case
            if index == len(word):
                if node.is_end > 0:
                    matching[prefix] = node.is_end
                return

            # if character is a period, recursively call on on all children
            if word[index] == '.':
                for num in range(0, 26):  # num + ord(a)   <--- integer chr()
                    if node.children[num] is not None:
                        autocomplete_inner(node.children[num], prefix + chr(num + 97), index + 1)

            # if character is not a period, call on child with next matching character of word/prefix
            elif node.get_child(word[index]) is not None:
                # Recursively call
                autocomplete_inner(node.get_child(word[index]), prefix + word[index], index + 1)

        autocomplete_inner(self.root, "", 0)
        return matching


class TrieClassifier:
    """
    Implementation of a trie-based text classifier.
    """

    # DO NOT MODIFY

    __slots__ = "tries"

    def __init__(self, classes: List[str]) -> None:
        """
        Constructs a TrieClassifier with specified classes.
        :param classes: List of possible class labels of training and testing data.
        :return: None.
        """
        self.tries = {}
        for cls in classes:
            self.tries[cls] = Trie()

    @staticmethod
    def accuracy(labels: List[str], predictions: List[str]) -> float:
        """
        Computes the proportion of predictions that match labels.
        :param labels: List of strings corresponding to correct class labels.
        :param predictions: List of strings corresponding to predicted class labels.
        :return: Float proportion of correct labels.
        """
        correct = sum([1 if label == prediction else 0 for label, prediction in zip(labels, predictions)])
        return correct / len(labels)

    # Implement Below

    def fit(self, class_strings: Dict[str, List[str]]) -> None:
        """
        Adds every individual word in a list of strings associated with each class to the Trie
        corresponding to the class in self.trie
        :param class_strings: A dictionary of (class, List[str]) pairs to train the classifier on.
        :return: None
        """
        for classes in class_strings:
            for each_sentence in class_strings[classes]:
                new_list = each_sentence.split()
                for individual_words in new_list:
                    self.tries[classes].add(individual_words)

    def predict(self, strings: List[str]) -> List[str]:
        """
        Predicts the class of a tweet.
        :param strings: List of tweets.
        :return: List of predicted classes corresponding to the input strings.
        """
        score_list = []
        for sentence in strings:  # each sentence/tweet
            max_score = 0
            best_class = ""
            single_words_list = sentence.split()
            for each_trie_class in self.tries:  # each class
                is_end_count = 0
                for single_words in single_words_list:  # each word in sentences
                    is_end_count += self.tries[each_trie_class].search(single_words)
                score = (is_end_count / len(self.tries[each_trie_class]))
                if score > max_score:
                    max_score = score
                    best_class = each_trie_class
            score_list.append(best_class)

        return score_list
