class Config(object):
    def __init__(self):
        self.expand_ratio = 2
        self.KV_size = 128
        self.transformer = {
            'num_layers': 2,
            'num_heads': 4,
            'hidden_size': 128,
            'mlp_dim': 512,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
        }
