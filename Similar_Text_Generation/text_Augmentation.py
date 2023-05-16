import jieba
import synonyms
import random
from random import shuffle
random.seed(2019)
import argparse
paser = argparse.ArgumentParser()
paser.add_argument("--input",   required=True,    type=str,    help="原始数据的输入文件目录")
paser.add_argument("--output",  required=False,  type=str,  help="增强数据后的输出文件目录")
paser.add_argument("--num_aug", required=False, type=int, help="每条原始语句增强的语句数")
paser.add_argument("--alpha",   required=False,   type=float, help="每条语句中将会被改变的单词数占比")
paser.add_argument("--num_aug",   default= 9,   type=int, help="每条原始语句增强的语句数")
args = paser.parse_args()

class Augmentation:
        def __init__(self):
            self.stop_words = self._read_stop_word()
        def _read_stop_word():
          # 停用词列表，默认使用哈工大停用词表
          f = open('./stopwords/hit_stopwords.txt')
          stop_words = []
          for stop_word in f.readlines():
              stop_words.append(stop_word[:-1])
        ########################################################################
        # 同义词替换 -- 替换一个语句中的n个单词为其同义词
        ########################################################################
        def   synonym_replacement(self,words, n):
            new_words = words.copy()
            random_word_list = list(set([word for word in words if word not in stop_words]))
            random.shuffle(random_word_list)
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = self.get_synonyms(random_word)
                if len(synonyms) >= 1:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if word == random_word else word for word in new_words]
                    num_replaced += 1
                if num_replaced >= n:
                    break
            sentence = ' '.join(new_words)
            new_words = sentence.split(' ')
            return new_words

        def   get_synonyms(self,word):
             return synonyms.nearby(word)[0]

        ########################################################################
        # 随机插入 -- 随机在语句中插入n个词
        ########################################################################
        def random_insertion(self,words, n):
            new_words = words.copy()
            for _ in range(n):
                self.add_word(new_words)
            return new_words

        def add_word(self,new_words):
            synonyms = []
            counter = 0
            while len(synonyms) < 1:
                random_word = new_words[random.randint(0, len(new_words ) -1)]
                synonyms = self.get_synonyms(random_word)
                counter += 1
                if counter >= 10:
                    return
            random_synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(new_words ) -1)
            new_words.insert(random_idx, random_synonym)

        ########################################################################
        # Random swap -- Randomly swap two words in the sentence n times
        ########################################################################
        def random_swap(self,words, n):
            new_words = words.copy()
            for _ in range(n):
                new_words = self.swap_word(new_words)
            return new_words

        @staticmethod
        def swap_word(new_words):
            random_idx_1 = random.randint(0, len(new_words ) -1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_words ) -1)
                counter += 1
                if counter > 3:
                    return new_words
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
            return new_words

        ########################################################################
        # 随机删除 -- 以概率p删除语句中的词
        ########################################################################
        @staticmethod
        def random_deletion(words, p):
            if len(words) == 1:
                return words
            new_words = []
            for word in words:
                r = random.uniform(0, 1)
                if r > p:
                    new_words.append(word)
            if len(new_words) == 0:
                rand_int = random.randint(0, len(words) -1)
                return [words[rand_int]]
            return new_words

model = Augmentation()
########################################################################
# EDA函数
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)
    augmented_sentences = []
    num_new_per_technique = int(num_aug) +1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = model.synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    # 随机插入ri
        a_words = model.random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
    # 随机交换rs
        a_words = model.random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))
    # 随机删除rd
        a_words = model.random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))
    # print(augmented_sentences)
    shuffle(augmented_sentences)
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    augmented_sentences.append(seg_list)
    return augmented_sentences

def gen_eda(train_orig, output_file, alpha, num_aug=9):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    print("正在使用EDA生成增强语句...")
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')  # 使用[:-1]是把\n去掉了
        sentence1 = parts[0]
        sentence2 = parts[1]
        label = parts[2]
        aug_sentences1 = eda(sentence1, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        aug_sentences2 = eda(sentence2, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for i ,aug_sentence1 in enumerate(aug_sentences1):
            writer.write(aug_sentence1.replace(' ' ,'') + "\t" + aug_sentences2[i].replace(' ' ,'') + "\t" + label +'\n')
    writer.close()
    print("已生成增强语句!")
    print(output_file)


if __name__ == "__main__":
    gen_eda(args.input, output, alpha=args.alpha, num_aug=args.num_aug)