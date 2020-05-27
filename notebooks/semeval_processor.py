import csv
import sys
from io import open
import numpy as np

#Tweaked Huggingface DataProcessor code:
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e11_p, e12_p, e21_p, e22_p,
                 e1_mask, e2_mask,
                 segment_ids,
                 label_id):
                 
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.e11_p = e11_p
        self.e12_p = e12_p
        self.e21_p = e21_p
        self.e22_p = e22_p
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.segment_ids=segment_ids
        self.label_id = label_id

        
class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()


    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
        
        
class SemEvalProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(
                self._read_tsv(input_file), "train")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(
                self._read_tsv(input_file), "test")
        
    def get_inference(self, input_str):
        """converts input string into InputExample object"""
        out = ['0', input_str]
        out_list = [out]
        return self._create_examples(out_list, "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(19)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,: 
        2	the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .	6
        """
        examples = []
        for line in lines:
            guid = line[0]
            text_a = line[1]
            text_b = None
            if set_type == "test":
                label = 0
            else:
                label = line[2]
                
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

#Class for RElation extraction encoding/decoding in SemEval standard
class Semeval():
    
    def __init__(self):
        self.processor = SemEvalProcessor()
        self.RELATION_LABELS = ['Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
                   'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                   'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                   'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                   'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                   'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                   'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                   'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                   'Content-Container(e1,e2)', 'Content-Container(e2,e1)']
        
        
    def re_encode(self, input_text,tokenizer,max_seq_len=128):
        
        tokenizer.add_special_tokens({'additional_special_tokens':
                                    ['<e1>', '</e1>','<e2>','</e2>']})
        input_example = self.processor.get_inference(input_text)
        
        input_features = self.convert_examples_to_features(examples=input_example,
                                            max_seq_len=max_seq_len,
                                            tokenizer=tokenizer,
                                            return_tensors=False)
        
        return [input_features['input_ids'],
                input_features['attention_mask'],
                input_features['e1_mask'],
                input_features['e2_mask']]
    
    def re_decode(self,prediction):
        
        return self.RELATION_LABELS[prediction.argmax()]    
    
    
    def convert_examples_to_features(self, examples, max_seq_len,
                                     tokenizer, 
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True,
                                     use_entity_indicator=True,
                                     return_tensors=True):
        """does this"""

        features = []
        for example in examples:

            tokens_a = tokenizer.tokenize(example.text_a)

            # Account for [CLS] and [SEP] with "- 2"
            special_tokens_count = 2
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = self._truncate_seq(tokens_a, max_seq_len - special_tokens_count)

            tokens = [cls_token] + tokens_a + [sep_token]
            segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens)-1)

            assert len(tokens) == len(segment_ids)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # entity mask
            if use_entity_indicator:
                if "</e2>" not in tokens or "</e1>" not in tokens:  # remove this sentence because after max length truncation, the one entity boundary is broken
                    continue 
                else:
                    e11_p = tokens.index("<e1>")+1
                    e12_p = tokens.index("</e1>")
                    e21_p = tokens.index("<e2>")+1
                    e22_p = tokens.index("</e2>")

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)
            if use_entity_indicator:
                e1_mask = [0 for i in range(len(input_mask))]
                e2_mask = [0 for i in range(len(input_mask))]
                for i in range(e11_p, e12_p):
                    e1_mask[i] = 1
                for i in range(e21_p, e22_p):
                    e2_mask[i] = 1

            assert len(input_ids) == max_seq_len, f"Error in sample: {example.guid}, len(input_ids)={len(input_ids)}"
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            label_id = example.label

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              e11_p=e11_p,
                              e12_p=e12_p,
                              e21_p=e21_p,
                              e22_p=e22_p,
                              e1_mask=e1_mask,
                              e2_mask=e2_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

        if return_tensors:
            
            from tf import Dataset

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, 
                            "attention_mask": ex.input_mask, 
                            "token_type_ids": ex.segment_ids,
                            "e1_mask": ex.e1_mask,
                            "e2_mask": ex.e2_mask},
                            ex.label_id
                            )

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, 
                "attention_mask": tf.int32, 
                "token_type_ids": tf.int32,
                "e1_mask": tf.int32,
                "e2_mask": tf.int32}, 
                tf.int64),
                ({"input_ids": tf.TensorShape([None]), 
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
                "e1_mask": tf.TensorShape([None]),
                "e2_mask": tf.TensorShape([None])}, 
                tf.TensorShape([])),
            )
            return dataset

        else:
            input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, label_id = [],[],[],[],[],[]
            for ex in features:
                input_ids.append(ex.input_ids)
                attention_mask.append(ex.input_mask)
                token_type_ids.append(ex.segment_ids)
                e1_mask.append(ex.e1_mask)
                e2_mask.append(ex.e2_mask)
                label_id.append(ex.label_id)

            return {
                    "input_ids": np.asarray(input_ids, dtype='int32'),
                    "attention_mask": np.asarray(attention_mask, dtype='int32'),
                    "token_type_ids": np.asarray(token_type_ids, dtype='int32'),
                    "e1_mask": np.asarray(e1_mask, dtype='int32'),
                    "e2_mask": np.asarray(e2_mask, dtype='int32'),
                    "label_id": np.asarray(label_id, dtype='int32')
            }


    def _truncate_seq(self, tokens_a, max_length):
        """Truncates a sequence """
        tmp = tokens_a[:max_length]
        if ("[E12]" in tmp) and ("[E22]" in tmp):
            return tmp
        else:
            e11_p = tokens_a.index("[E11]")
            e12_p = tokens_a.index("[E12]")
            e21_p = tokens_a.index("[E21]")
            e22_p = tokens_a.index("[E22]")
            start = min(e11_p, e12_p, e21_p, e22_p)
            end = max(e11_p, e12_p, e21_p, e22_p)
            if end-start > max_length:
                remaining_length = max_length - (e12_p-e11_p+1) - (e22_p-e21_p+1)  
                first_addback = math.floor(remaining_length/2)
                second_addback = remaining_length - first_addback
                if start == e11_p:
                    new_tokens = tokens_a[e11_p: e12_p+1+first_addback] + tokens_a[e21_p-second_addback:e22_p+1]
                else:
                    new_tokens = tokens_a[e21_p: e22_p+1+first_addback] + tokens_a[e11_p-second_addback:e12_p+1]
                return new_tokens
            else:
                new_tokens = tokens_a[start:end+1]
                remaining_length = max_length - len(new_tokens)
                if start < remaining_length:  # add sentence beginning back
                    new_tokens = tokens_a[:start] + new_tokens 
                    remaining_length -= start
                else:
                    new_tokens = tokens_a[start-remaining_length:start] + new_tokens
                    return new_tokens

                # still some room left, add sentence end back
                new_tokens = new_tokens + tokens_a[end+1:end+1+remaining_length]
                return new_tokens
