import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    #  对于给定的unigram/bigram/trigram字符串
    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        # 最后一个char的is_word标识置为True，表示查找到该处停止
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            # 没有找到
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    # word:char_list(以instance中不同index的char开头)
    # 寻找所有以word[0]开头的可以匹配到lexicon的str
    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’
        matched = []

        while len(word) > 0:
            if self.search(word):
                # 将word中每个元素以""相互连接起来，形成str
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

