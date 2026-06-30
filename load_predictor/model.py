from transformers import AutoConfig, AutoTokenizer, AutoModel, DistilBertModel, T5Tokenizer
import torch
import torch.nn as nn


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


class ResponsePredictor(nn.Module):
    def __init__(self, model_name="distilbert/distilbert-base-uncased", response_type=1, hidden_dim=512, use_prompt=1, prompt_learning_length=12):
        super().__init__()
        self.model_name = model_name
        self.response_type = response_type
        self.use_prompt = use_prompt
        self.hidden_dim = hidden_dim
        self.prompt_learning_length = prompt_learning_length
        
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert = DistilBertModel.from_pretrained(self.model_name)
        
        
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False
            
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_grad = True
        
        self.cls = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            QuickGELUActivation(),
            nn.Linear(self.hidden_dim, self.response_type),
        )
        
        self.leanrable_prompts = nn.Parameter(torch.empty(self.prompt_learning_length, self.config.hidden_size), requires_grad=True)
        nn.init.normal_(self.leanrable_prompts, std=0.01)

        replace_str = ''
        for i in range(self.prompt_learning_length):
            replace_str += '.'
        self.replace_str = replace_str
        self.prompt_learning_length = self.prompt_learning_length


    def forward(self, prompts, device):
        if self.use_prompt:
            prompts = [self.replace_str + prompt for i, prompt in enumerate(prompts)]
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
            word_embedding = self.bert.embeddings.word_embeddings(inputs.input_ids)
            word_embedding[:, 1:self.prompt_learning_length+1] = self.leanrable_prompts
            if "distil" in self.model_name:
                inputs_embeds = self.bert.embeddings(input_ids=None, input_embeds=word_embedding)
            else:
                inputs_embeds = self.bert.embeddings(inputs_embeds=word_embedding)
            outputs = self.bert(attention_mask=inputs.attention_mask, inputs_embeds=inputs_embeds)
        
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        else:
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = self.bert(**inputs)

        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls(logits)
       
        return output
