import semeval_processor
from io import open
import pandas as pd

def features_from_tsv(tsv_file,tokenizer,test=False,max_seq_len=128):
    """creates train and test features from tsv files.
    Args:
        tsv_file: input tsv file with strings with id, text, label (if test=False) 
                    (eg.: 8 \t <e1>People</e1> have been moving back into <e2>downtown</e2>. \t 7 )
        tokenizer: add from pre-trained models (e.g. BertTokenizer.from_pretrained('bert-base-uncased'))
        
    Returns:
        features with the following structure:
            features['input_ids'],
            features['attention_mask'],
            features['e1_mask'],
            features['e2_mask'],
            features['label_id']
    """
    
    processor = semeval_processor.SemEvalProcessor()
    encoder = semeval_processor.Semeval()
    tokenizer.add_special_tokens({'additional_special_tokens':
                                ['<e1>', '</e1>','<e2>','</e2>']})
    
    if test:
        examples = processor.get_test_examples(tsv_file)        
    else:
        examples = processor.get_train_examples(tsv_file)

    features = encoder.convert_examples_to_features(examples= examples,
                                                    max_seq_len=max_seq_len,
                                                    tokenizer=tokenizer,
                                                    return_tensors=False)
    return features


def format_semeval_inputs(input_file, labels=None):

    with open(input_file) as raw_text:

        data = []
        lines = [line.strip() for line in raw_text]

        if labels: #train text:

            for idx in range(0, len(lines), 4):

                id = lines[idx].split("\t")[0]
                relation = labels.index(lines[idx + 1])
                sentence = lines[idx].split("\t")[1][1:-1]

                data.append([id, sentence, relation])
            
            return pd.DataFrame(data=data, columns=["id", "sentence", "relation"])

        else: #test text:
            
            for idx in range(0, len(lines)):

                id = lines[idx].split("\t")[0]
                sentence = lines[idx].split("\t")[1][1:-1]

                data.append([id, sentence])

            return pd.DataFrame(data=data, columns=["id", "sentence"])
        
def semeval_make_tsv(input_file, output_file, labels=None):

    format_semeval_inputs(input_file, labels).to_csv(output_file, sep='\t', index=False, header=False)
    