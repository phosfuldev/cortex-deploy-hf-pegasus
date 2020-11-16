import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

class PythonPredictor:
    def __init__(self, config):
        self.model_name = 'google/pegasus-reddit_tifu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"using device: {self.device}")
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name, force_download=True)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name, force_download=True).to(self.device)
        # self.tokenizer = PegasusTokenizer.from_pretrained('./model')
        # self.model = PegasusForConditionalGeneration.from_pretrained('./model').to(self.device)
    
    def predict(self, payload):
        text = payload["text"]
        print('\nsource text: {}\n'.format(text))
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        tokens = self.model.generate(input_ids)
        summary = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        print('\ngenerated summary: {}\n'.format(summary))
        return summary[0]