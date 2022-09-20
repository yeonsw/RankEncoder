import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead

from transformers.modeling_outputs import SequenceClassifierOutput


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    x_rank = x.argsort(dim=1)
    ranks = torch.zeros_like(x_rank, dtype=torch.float)
    n, d = x_rank.size()

    for i in range(n):
        ranks[i][x_rank[i]] = torch.arange(d, dtype=torch.float).to(ranks.device)
    return ranks

def cal_spr_corr(x: torch.Tensor, y: torch.Tensor):
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    x_rank_mean = torch.mean(x_rank, dim=1).unsqueeze(1)
    y_rank_mean = torch.mean(y_rank, dim=1).unsqueeze(1)
    xn = x_rank - x_rank_mean
    yn = y_rank - y_rank_mean
    x_var = torch.sqrt(torch.sum(torch.square(xn), dim=1).unsqueeze(1))
    y_var = torch.sqrt(torch.sum(torch.square(yn), dim=1).unsqueeze(1))
    xn = xn / x_var
    yn = yn / y_var

    return torch.mm(xn, torch.transpose(yn, 0, 1))

def cl_init(cls, config, model_loss=None):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    distances1=None,
    distances2=None,
    baseE_vecs1=None,
    baseE_vecs2=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    torch.cuda.empty_cache()

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )

    # Obtain sentence embeddings from [MASK] token
    index = input_ids == cls.mask_token_id
    last_hidden_state = outputs.last_hidden_state[index]
    assert last_hidden_state.size() == torch.Size([batch_size * num_sent, last_hidden_state.size(-1)])

    # During training, add an extra MLP layer with activation function
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(last_hidden_state)
    else:
        pooler_output = last_hidden_state

    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))
    return pooler_output, last_hidden_state
    
class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, mask_token_id=None, alpha=None, beta=None, lambda_=None, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        self.mask_token_id = mask_token_id

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

        self.alpha = alpha  # 0.1
        self.beta = beta  # 0.3
        self.lambda_ = lambda_  # 1e-3

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        distances1=None,
        distances2=None,
        baseE_vecs1=None,
        baseE_vecs2=None,
    ):
        return cl_forward(self,
                          self.bert,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict,
                          mlm_input_ids=mlm_input_ids,
                          mlm_labels=mlm_labels,
                          distances1=distances1,
                          distances2=distances2,
                          baseE_vecs1=baseE_vecs1,
                          baseE_vecs2=baseE_vecs2,
                        )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, mask_token_id, alpha=None, beta=None, lambda_=None, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mask_token_id = mask_token_id

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

        self.alpha = alpha  # 0.1
        self.beta = beta  # 0.3
        self.lambda_ = lambda_  # 5e-4

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        distances1=None,
        distances2=None,
        baseE_vecs1=None,
        baseE_vecs2=None,
    ):

        return cl_forward(self, self.roberta,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict,
                          mlm_input_ids=mlm_input_ids,
                          mlm_labels=mlm_labels,
                          distances1=distances1,
                          distances2=distances2,
                          baseE_vecs1=baseE_vecs1,
                          baseE_vecs2=baseE_vecs2,
                        )
