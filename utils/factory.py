from models.finetune import Finetune
from models.fa import FA

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        return Finetune(args)
    elif name == "fa":
        return FA(args)
    else:
        assert 0
