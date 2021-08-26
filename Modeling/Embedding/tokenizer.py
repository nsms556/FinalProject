import sentencepiece as spm


class TitleTokenizer:
    def __init__(self):
        pass
    
    def make_input_file(self, input_fn, sentences):
        with open(input_fm, 'w', encoding='utf8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
    
    def train_tokenizer(self, input_fn, prefix, vocab_size, model_type):
        templates = '--input={} --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --model_prefix={} --vocab_size={} --character_coverage=1.0 --model_type={}'
        cmd = templates.format(
            input_fn,
            prefix,
            vocab_size,
            model_type
        )
        
        spm.SentencePieceTrainer.Train(cmd)
        print(f"tokenizer model {prefix}.model is trained")
    
    def get_tokens(self, sp, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokens = sp.EncodeAsPieces(sentence)
            new_tokens = []
            for token in tokens:
                token = token.replace("_", "")
                if len(token) > 1:
                    new_tokens.append(token)
            if len(new_tokens) > 1:
                tokenized_sentences.append(new_tokens)
        return tokenized_sentences
