import pandas as pd
from processors.utils import InputExample
from model import tokenization_albert
from tqdm import tqdm

# id2label = {0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技', 5: '时尚', 6: '时政', 7: '游戏', 8: '娱乐', 9: '体育'}
id2risk = {0: '高风险', 1: '中风险', 2: '可公开', 3: '低风险', 4: '中风险', 5: '低风险', 6: '高风险', 7: '低风险', 8: '可公开', 9: '可公开'}
label2risk = {'财经': '高风险', '房产': '中风险', '家居': '可公开', '教育': '低风险', '科技': '中风险', '时尚': '低风险', '时政': '高风险', '游戏': '低风险',
              '娱乐': '可公开', '体育': '可公开'}
label2id = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4, '时尚': 5, '时政': 6, '游戏': 7, '娱乐': 8, '体育': 9}


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def get_all_stopword():
    words_set = set()
    # path_list = ['baidu', 'cn', 'hit', 'scu', 'news']
    path_list = ['baidu', 'cn', 'hit', 'scu']
    for i in path_list:
        path = './stopword/' + i + '_stopwords.txt'
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        for _word in data:
            words_set.add(_word.strip())
    return words_set


def get_labeled_data_from_csv(path):
    inputexample_list = []
    df = pd.read_csv(path, encoding='utf-8')
    for index, row in df.iterrows():
        inputexample_list.append(
            InputExample(guid=row["id"], text_a=row["content"].strip().replace(' ', ''), label=row["class_label"]))
    return inputexample_list


def get_unlabeled_data_from_csv(path):
    inputexample_list = []
    df = pd.read_csv(path, encoding='utf-8')
    for index, row in df.iterrows():
        inputexample_list.append(
            InputExample(guid=row["id"], text_a=row["content"].strip().replace(' ', '')))
    return inputexample_list


def get_train_examples(flag='original'):
    if flag == 'original':
        path = './finetunedata/labeled_data.csv'
    elif flag == 'extract':
        path = 'finetunedata/labeled_data_extract_clean.csv'
    else:
        path = ''
    assert path != ''
    # path = './finetunedata/train_data.csv'
    print('train data path', path)
    return get_labeled_data_from_csv(path)


def get_dev_examples(flag='original'):
    if flag == 'original':
        path = './finetunedata/unlabeled_data_clean.csv'
    elif flag == 'extract':
        path = 'finetunedata/unlabeled_data_labeled_extract_clean.csv'
    else:
        path = ''
    assert path != ''
    print('dev data path', path)
    return get_unlabeled_data_from_csv(path)


def get_test_examples(flag='original'):
    if flag == 'original':
        path = './finetunedata/test_data_clean.csv'
    elif flag == 'extract':
        path = 'finetunedata/test_data_extract_clean.csv'
    else:
        path = ''
    assert path != ''
    print('test data path', path)
    return get_unlabeled_data_from_csv(path)


def get_all_example(flag='extract'):
    train_example = get_train_examples(flag)
    dev_example = get_dev_examples(flag)
    test_example = get_test_examples(flag)
    all_example = []
    all_example.extend(train_example)
    all_example.extend(dev_example)
    all_example.extend(test_example)
    return all_example


def get_unk_token():
    # 补全字典
    all_example = get_all_example()
    unk_dict = {}
    vocab_file = './prev_trained_model/vocab.txt'
    tokenizer = tokenization_albert.FullTokenizer(vocab_file=vocab_file, do_lower_case=True, spm_model_file=None)
    for example in tqdm(all_example):
        for word in tokenizer.tokenize(example.text_a):
            if word in unk_dict:
                unk_dict[word] += 1
            else:
                unk_dict[word] = 1

    print(len(unk_dict))
    dict_sort = sorted(unk_dict.items(), key=lambda x: x[1], reverse=True)
    # print(dict_sort)
    with open('./prev_trained_model/unk_vocab.txt', 'w', encoding='utf-8') as f:
        for i in dict_sort:
            f.write(str(i) + '\n')


def get_unclean_token():
    # 获取纯英文、纯中文、纯数字之外的字符
    other_dict = {}
    all_example = get_all_example()
    tokenizer = tokenization_albert.BasicTokenizer(do_lower_case=True)
    for example in tqdm(all_example):
        tokens = tokenizer.tokenize(example.text_a)
        for token in tokens:
            if is_chinese(token) or str.isdigit(token) or str.isalpha(token):
                pass
            else:
                if token in other_dict:
                    other_dict[token] += 1
                else:
                    other_dict[token] = 1

    print(len(other_dict))
    dict_sort = sorted(other_dict.items(), key=lambda x: x[1], reverse=True)
    # print(dict_sort)
    with open('./prev_trained_model/unclean_token.txt', 'w', encoding='utf-8') as f:
        for i in dict_sort:
            f.write(str(i) + '\n')


def to_standard_train_data():
    path = './finetunedata/labeled_data_clean.csv'
    cont_list = []
    df = pd.read_csv(path, encoding='utf-8')
    for index, row in df.iterrows():
        # print(row["class_label"])
        cont_list.append(
            {'id': row["id"], 'class_label': row["class_label"], 'content': row["content"].strip().replace(' ', '')})
    df_s = pd.DataFrame(cont_list, columns=['id', 'class_label', 'content'])
    df_s.to_csv('./finetunedata/labeled_data_clean_1.csv', columns=['id', 'class_label', 'content'], index=False,
                header=True,
                encoding='utf-8')
    print("end")
    print(df_s.columns)


def load_train_dev_data():
    path_train = './finetunedata/labeled_data_clean.csv'
    path_dev = './finetunedata/valid_uns_new.csv'
    # path_train = './finetunedata/train_uns_fair.csv'
    # path_dev = './finetunedata/valid_uns_fair.csv'
    # path_train = './finetunedata/train_dataset5.csv'
    # path_dev = './finetunedata/valid_dataset5.csv'
    inputexample_train = get_labeled_data_from_csv(path_train)
    inputexample_valid = get_labeled_data_from_csv(path_dev)
    # print(len(inputexample_train))
    # print(len(inputexample_valid))
    return inputexample_train, inputexample_valid


if __name__ == '__main__':
    # print(get_train_examples())
    # print(get_test_examples())
    # get_dev_examples()
    # get_unk_token()
    # get_unclean_token()
    # to_standard_train_data()
    load_train_dev_data()
