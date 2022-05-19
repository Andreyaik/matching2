import torch
from transformers import AutoModel, AutoTokenizer
from transformers import logging
logging.set_verbosity_error()


def model_matching(device=torch.device('cpu'),
                   trained_model='triplet_model_full_plus_e_v1.pt'):
    """The function loads the trained model and
    transfers it to the specified device
    :param device: available device
    :param trained_model: path of the trained model
    :return: object is a pre-trained model placed on a given device
    """
    model_nn = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    model_nn.to(device)
    if device == torch.device('cpu'):
        model_nn.load_state_dict(torch.load(trained_model,
                                            map_location=torch.device('cpu')))
    else:
        model_nn.load_state_dict(torch.load(trained_model))
    model_nn.eval()

    return model_nn


def out_model(model_trained, tokenized_title):
    """ function returns a vector of dimension 312 for
    the input product name
    :param model_trained: object of the pre-trained model
    :param tokenized_title: dictionary with a tokenized product name
    (in the dictionary 'input_ids', 'token_type_ids', 'attention_mask')
    :return: CLS vector: tensor of dimension (batch, 312)
    """
    output = model_trained(**{k: v.to(model_trained.device) for k, v in tokenized_title.items()})
    out = output.last_hidden_state[:, 0, :]
    return out


if __name__ == '__main__':
    model = model_matching()
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    t = tokenizer('Hi word', padding=True, truncation=True, return_tensors='pt')
    print(out_model(model, t).detach().cpu().numpy().shape)

