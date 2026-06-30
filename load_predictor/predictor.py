import sys as _sys, types as _types, logging as _logging

def _setup_llmserve_stubs():
    def make_pkg(name):
        if name in _sys.modules:
            return _sys.modules[name]
        m = _types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        _sys.modules[name] = m
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(make_pkg(parent), child, m)
        return m
    for mod in ["LLMServe","LLMServe.logger","LLMServe.global_scheduler",
                "LLMServe.global_scheduler.load_predictor",
                "LLMServe.global_scheduler.load_predictor.model"]:
        make_pkg(mod)
    _sys.modules["LLMServe.logger"].init_logger = lambda *a, **kw: _logging.getLogger()
    from load_predictor.model import ResponsePredictor, QuickGELUActivation
    m = _sys.modules["LLMServe.global_scheduler.load_predictor.model"]
    m.ResponsePredictor = ResponsePredictor
    m.QuickGELUActivation = QuickGELUActivation
    import transformers.models.distilbert.modeling_distilbert as _dm
    if not hasattr(_dm, "DistilBertSelfAttention"):
        _dm.DistilBertSelfAttention = _dm.MultiHeadSelfAttention
        if not hasattr(_dm, 'DistilBertSdpaAttention'):
            _dm.DistilBertSdpaAttention = _dm.MultiHeadSelfAttention

_setup_llmserve_stubs()

import torch
import os
import logging; init_logger = lambda: logging.getLogger(__name__)
from .model import ResponsePredictor


logger = init_logger()


class LoadPredictor:
    def __init__(self, scheduler_config):
        self.model_path = scheduler_config['req_predictor_model_path']
        if not os.path.exists(self.model_path):
            logger.error(f"Model path {self.model_path} does not exist!")
            raise FileNotFoundError(f"Model path {self.model_path} does not exist!")        
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.model = ResponsePredictor().to(self.device)
        self.load_model()

    def load_model(self):
        # Alias renamed class so pickle can find it
        import transformers.models.distilbert.modeling_distilbert as _dm
        if not hasattr(_dm, 'DistilBertSelfAttention'):
            _dm.DistilBertSelfAttention = _dm.MultiHeadSelfAttention
            if not hasattr(_dm, 'DistilBertSdpaAttention'):
                _dm.DistilBertSdpaAttention = _dm.MultiHeadSelfAttention
        load_mdoel = torch.load(self.model_path,
            weights_only=False,
            map_location=self.device,
        )
        self.model.bert.transformer.layer[-1] = load_mdoel['last_layer']
        self.model.cls = load_mdoel['cls']
        self.model.leanrable_prompts = load_mdoel['prompt']
    
    def predict(self, text):
        prediction = self.model([text], self.device).unsqueeze(0)
        return int(prediction.item())
    