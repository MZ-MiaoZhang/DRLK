import pickle
from urllib import parse
import json
from tqdm import tqdm

import numpy as np
import torch
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    from transformers import AlbertTokenizer
except:
    pass
HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1
from modeling.modeling_lm import MODEL_NAME_TO_CLASS

def load_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length, format=[]):
    class InputExample(object):
        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                extra_args = {'add_prefix_space': True} if (
                    model_type in ['roberta'] and 'add_prefix_space' in format) else {}
                tokens_a = tokenizer.tokenize(context, **extra_args)   
                tokens_b = tokenizer.tokenize(
                    example.question + " " + ending, **extra_args) 

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(
                    tokens_a, tokens_b, max_seq_length - special_tokens_count) 

                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * \
                        (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
               
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_mask = [
                    1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids(
                    [cls_token, sep_token])
                
                output_mask = [
                    1 if id in special_token_id else 0 for id in input_ids]

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1]
                                  * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] *
                                   padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + \
                        ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + \
                        ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append(
                    (tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(
                example_id=example.example_id, choices_features=choices_features, label=label))

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

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(
            features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(
            features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(
            features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(
            features, 'output_mask'), dtype=torch.uint8)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer,
                       'roberta': RobertaTokenizer, }.get(model_type)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)

    if any(x in format for x in ('add_qa_prefix', 'fairseq')):
        for example in examples:
            example.contexts = ['Q: ' + c for c in example.contexts]
            example.endings = ['A: ' + e for e in example.endings]

    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            # xlnet has a cls token at the end
                                            cls_token_at_end=bool(
                                                model_type in ['xlnet']),
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']
                                                                 and 'no_extra_sep' not in format
                                                                 and 'fairseq' not in format),
                                            cls_token_segment_id=2 if model_type in [
                                                'xlnet'] else 0,
                                            # pad on the left for xlnet
                                            pad_on_left=bool(
                                                model_type in ['xlnet']),
                                            pad_token=tokenizer.pad_token_id or 0,
                                            pad_token_segment_id=4 if model_type in [
                                                'xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta'] else 1)

    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)

    return (example_ids, all_label, *data_tensors)

def load_adj_data(adj_pk_path, max_node_num, num_choice, emb_pk_path=None):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    adj_data = []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)  # node_numbers
    concept_ids = torch.zeros((n_samples, max_node_num), dtype=torch.long)   # node_true_id
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) # node_type question answer others

    if emb_pk_path is not None:
        with open(emb_pk_path, 'rb') as fin:
            all_embs = pickle.load(fin)
        emb_data = torch.zeros(
            (n_samples, max_node_num, all_embs[0].shape[1]), dtype=torch.float)

    adj_lengths_ori = adj_lengths.clone()
    for idx, (_data) in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
        adj, concepts, qm, am = _data['adj'], _data['concepts'], _data['qmask'], _data['amask']
        num_concept = min(len(concepts), max_node_num)
        adj_lengths_ori[idx] = len(concepts)
        if emb_pk_path is not None:
            embs = all_embs[idx]
            assert embs.shape[0] >= num_concept
            emb_data[idx, :num_concept] = torch.tensor(embs[:num_concept])
            concepts = np.arange(num_concept)
        else:
            concepts = concepts[:num_concept]
        concept_ids[idx, :num_concept] = torch.tensor(
            concepts)  # note : concept zero padding is disabled

        adj_lengths[idx] = num_concept
        node_type_ids[idx, :num_concept][torch.tensor(
            qm, dtype=torch.uint8)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(
            am, dtype=torch.uint8)[:num_concept]] = 1
        ij = torch.tensor(adj.row, dtype=torch.int64)
        k = torch.tensor(adj.col, dtype=torch.int64)
        if('norel' in adj_pk_path):
            n_node = adj.shape[1]
            mask = (ij < max_node_num) & (k < max_node_num)
            ij, k = ij[mask], k[mask]
            adj_data.append((ij, k))
        else:
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node
            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  
            adj_data.append((i, j, k))

    print('| ori_adj_len: {:.2f} | adj_len: {:.2f} |'.format(adj_lengths_ori.float().mean().item(), adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    concept_ids, node_type_ids, adj_lengths = [
        x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, adj_lengths)]
    if emb_pk_path is not None:
        emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])

    if('norel' in adj_pk_path):
        adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))
        if emb_pk_path is None:
            return concept_ids, node_type_ids, adj_lengths, adj_data, 0
        return concept_ids, node_type_ids, adj_lengths, emb_data, adj_data, 0
    else:
        adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))
        if emb_pk_path is None:
            return concept_ids, node_type_ids, adj_lengths, adj_data, half_n_rel * 2 + 1
        return concept_ids, node_type_ids, adj_lengths, emb_data, adj_data, half_n_rel * 2 + 1

class BatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_empty=None, adj_data=None):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_empty = adj_empty.to(self.device)
        self.adj_data = adj_data

    def clip_batch(self, batch):
        input_ids, attention_mask, token_type_ids, output_mask = batch
        batch_size = input_ids.size(0)
        while True:
            end_flag = False
            for i in range(batch_size):
                if input_ids[i, 0, -1] != 0:
                    end_flag = True
                if input_ids[i, 1, -1] != 0:
                    end_flag = True

            if end_flag:
                break
            else:
                input_ids = input_ids[:, :, :-1]

        max_seq_length = input_ids.size(2)

        attention_mask = attention_mask[:, :, :max_seq_length]
        token_type_ids = token_type_ids[:, :, :max_seq_length]
        output_mask = output_mask[:, :, :max_seq_length]

        return [input_ids, attention_mask, token_type_ids, output_mask]

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty
        batch_adj[:] = 0
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(
                self.labels[batch_indexes], self.device)
            batch_tensors0 = [self._to_device(
                x[batch_indexes], self.device) for x in self.tensors0]
            batch_tensors1 = [self._to_device(
                x[batch_indexes], self.device) for x in self.tensors1]

            batch_tensors0 = self.clip_batch(batch_tensors0)
            batch_lists0 = [self._to_device(
                [x[i] for i in batch_indexes], self.device) for x in self.lists0]
            batch_lists1 = [self._to_device(
                [x[i] for i in batch_indexes], self.device) for x in self.lists1]

            batch_adj[:] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, batch_adj[:b - a]])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)



class DRLK_DataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 train_embs_path=None, dev_embs_path=None, test_embs_path=None,
                 is_inhouse=False, inhouse_train_qids_path=None, use_contextualized=False,
                 subsample=1.0, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)

        num_choice = self.train_encoder_data[0].size(1)
        *self.train_decoder_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, num_choice, emb_pk_path=train_embs_path if use_contextualized else None)
        *self.dev_decoder_data, self.dev_adj_data, n_rel = load_adj_data(dev_adj_path, max_node_num, num_choice, emb_pk_path=dev_embs_path if use_contextualized else None)
        assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        # pre-allocate an empty batch adj matrix
        self.adj_empty = torch.zeros((self.batch_size, num_choice, n_rel - 1, max_node_num, max_node_num), dtype=torch.float32)
        self.eval_adj_empty = torch.zeros((self.eval_batch_size, num_choice, n_rel - 1, max_node_num, max_node_num), dtype=torch.float32)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
            *self.test_decoder_data, self.test_adj_data, n_rel = load_adj_data(test_adj_path, max_node_num, num_choice, emb_pk_path=test_embs_path if use_contextualized else None)
            assert all(len(self.test_qids) == len(self.test_adj_data) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def get_node_feature_dim(self):
        return self.train_decoder_data[-1].size(-1) if self.use_contextualized else None

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device0, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.adj_empty, adj_data=self.train_adj_data)

    def train_eval(self):
        return BatchGenerator(self.device0, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)

    def dev(self):
        return BatchGenerator(self.device0, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                             tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device0, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                                 tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)
        else:
            return BatchGenerator(self.device0, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
                                                 tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.test_adj_data)
