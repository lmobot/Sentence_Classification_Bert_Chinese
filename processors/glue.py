import logging
import torch
from .utils import InputFeatures, InputFeatures_split

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def collate_fn_Siamese(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens_a, all_lens_b, all_labels = map(torch.stack,
                                                                                                    zip(*batch))
    max_len_a = max(all_lens_a).item()
    max_len_b = max(all_lens_b).item()
    if max_len_b == 2:
        all_input_ids = all_input_ids[:, :max_len_a]
        all_attention_mask = all_attention_mask[:, :max_len_a]
        all_token_type_ids = all_token_type_ids[:, :max_len_a]
    else:
        all_input_ids = all_input_ids[:, :512 + max_len_b]
        all_attention_mask = all_attention_mask[:, :512 + max_len_b]
        all_token_type_ids = all_token_type_ids[:, :512 + max_len_b]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def collate_fn_predict(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids


def collate_fn_predict_Siamese(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens_a, all_lens_b = map(torch.stack,
                                                                                        zip(*batch))
    max_len_a = max(all_lens_a).item()
    max_len_b = max(all_lens_b).item()
    if max_len_b == 2:
        all_input_ids = all_input_ids[:, :max_len_a]
        all_attention_mask = all_attention_mask[:, :max_len_a]
        all_token_type_ids = all_token_type_ids[:, :max_len_a]
    else:
        all_input_ids = all_input_ids[:, :512 + max_len_b]
        all_attention_mask = all_attention_mask[:, :512 + max_len_b]
        all_token_type_ids = all_token_type_ids[:, :512 + max_len_b]
    return all_input_ids, all_attention_mask, all_token_type_ids


def ccf_convert_examples_to_features(examples, tokenizer,
                                     max_seq_length=512,
                                     label_list=None):
    """
    ccf_classification
    面向数据安全治理的数据内容智能发现与分级分类
    no tokens_b
    """

    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
        logger.info('label map %s', label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # tokens_a = tokens_a[100:]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if label_list:
            label_id = label_map[example.label]
            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label_id,
                              input_len=input_len))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              input_len=input_len))
    return features


def ccf_convert_examples_to_Siamese_features(examples, tokenizer,
                                             max_seq_length=512,
                                             label_list=None):
    """
    ccf_classification
    面向数据安全治理的数据内容智能发现与分级分类
    no tokens_b
    split text_a to two sentences
    """
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
        logger.info('label map %s', label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_temp = tokenizer.tokenize(example.text_a)
        tokens_b_temp = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a_temp) > max_seq_length - 2:
            tokens_b_temp = tokens_a_temp[(max_seq_length - 2):2 * (max_seq_length - 2)]
            tokens_a_temp = tokens_a_temp[0:(max_seq_length - 2)]

        tokens_a = []
        token_type_ids_a = []
        tokens_a.append("[CLS]")
        token_type_ids_a.append(0)
        for token in tokens_a_temp:
            tokens_a.append(token)
            token_type_ids_a.append(0)
        tokens_a.append("[SEP]")
        token_type_ids_a.append(0)

        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask_a = [1] * len(input_ids_a)
        input_len_a = len(input_ids_a)

        # Zero-pad up to the sequence length.
        while len(input_ids_a) < max_seq_length:
            input_ids_a.append(0)
            attention_mask_a.append(0)
            token_type_ids_a.append(0)

        assert len(input_ids_a) == max_seq_length
        assert len(attention_mask_a) == max_seq_length
        assert len(token_type_ids_a) == max_seq_length

        if tokens_b_temp:
            tokens_b = []
            token_type_ids_b = []
            tokens_b.append("[CLS]")
            token_type_ids_b.append(0)
            for token in tokens_b_temp:
                tokens_b.append(token)
                token_type_ids_b.append(0)
            tokens_b.append("[SEP]")
            token_type_ids_b.append(0)
        else:
            tokens_b = []
            token_type_ids_b = []
            tokens_b.append("[CLS]")
            token_type_ids_b.append(0)
            tokens_b.append("[SEP]")
            token_type_ids_b.append(0)

        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask_b = [1] * len(input_ids_b)
        input_len_b = len(input_ids_b)

        # Zero-pad up to the sequence length.
        while len(input_ids_b) < max_seq_length:
            input_ids_b.append(0)
            attention_mask_b.append(0)
            token_type_ids_b.append(0)

        assert len(input_ids_b) == max_seq_length
        assert len(attention_mask_b) == max_seq_length
        assert len(token_type_ids_b) == max_seq_length

        input_ids_a.extend(input_ids_b)
        attention_mask_a.extend(attention_mask_b)
        token_type_ids_a.extend(token_type_ids_b)

        all_max_seq_length = max_seq_length * 2
        assert len(input_ids_a) == all_max_seq_length
        assert len(attention_mask_a) == all_max_seq_length
        assert len(token_type_ids_a) == all_max_seq_length

        # feature可以切成两个放，这里使用一个
        if label_list:
            label_id = label_map[example.label]
            features.append(InputFeatures_split(input_ids=input_ids_a,
                                                attention_mask=attention_mask_a,
                                                token_type_ids=token_type_ids_a,
                                                label=label_id,
                                                input_len_a=input_len_a,
                                                input_len_b=input_len_b))
        else:
            features.append(InputFeatures_split(input_ids=input_ids_a,
                                                attention_mask=attention_mask_a,
                                                token_type_ids=token_type_ids_a,
                                                input_len_a=input_len_a,
                                                input_len_b=input_len_b))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
