
import warnings

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel,BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertModel,BertPredictionHeadTransform
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput, \
    QuestionAnsweringModelOutput, TokenClassifierOutput

from models.fusion_embedding import FusionBertEmbeddings
from models.modeling_glycebert import GlyceBertModel
from datasets.utils import Pinyin

SMALL_CONST = 1e-15

def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)


class Pinyin_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.classifier=nn.Linear(config.hidden_size, 1378)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        scores = self.classifier(sequence_output)
        return scores
    

class Phonetic_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pinyin=Pinyin()
        self.transform = BertPredictionHeadTransform(config)
        self.sm_classifier=nn.Linear(config.hidden_size,self.pinyin.sm_size)
        self.ym_classifier=nn.Linear(config.hidden_size,self.pinyin.ym_size)
        self.sd_classifier=nn.Linear(config.hidden_size,self.pinyin.sd_size)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sm_scores = self.sm_classifier(sequence_output)
        ym_scores = self.ym_classifier(sequence_output)
        sd_scores = self.sd_classifier(sequence_output)
        return sm_scores,ym_scores,sd_scores


class MultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Phonetic_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        sm_scores, ym_scores, sd_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, sm_scores, ym_scores, sd_scores

class AblationHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Pinyin_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        pinyin_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, pinyin_scores

class GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.loss_fct = CrossEntropyLoss()

        self.bert2 = GlyceBertModel(config)
        self.cls2 = MultiTaskHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        pinyin_labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gamma=1,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0)*(input_ids != 101)*(input_ids != 102).long()
        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores, sm_scores,ym_scores,sd_scores = self.cls(sequence_output)

        masked_lm_loss = None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)

        phonetic_loss=None
        if pinyin_labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, pinyin_labels[...,0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[...,1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[...,2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss = (sm_loss+ym_loss+sd_loss)/3

        loss = None
        if masked_lm_loss is not None :
            loss = masked_lm_loss 
        
        if not return_dict:
            output = (prediction_scores, sm_scores,ym_scores,sd_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_a,
        hidden_states_b,
        attention_mask=None,
        output_attentions=False,
    ):
        query_layer = self.query(hidden_states_a)
        key_layer = self.transpose_for_scores(self.key(hidden_states_b))
        value_layer = self.transpose_for_scores(self.value(hidden_states_b))

        query_layer = self.transpose_for_scores(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = context_layer

        return outputs


class Dynamic_GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(Dynamic_GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.net_gate = nn.Linear(config.hidden_size, 2)
        self.net_copy_gate = nn.Linear(config.hidden_size, 2)
        self.net_retrieve_gate = nn.Linear(config.hidden_size, 2)
        self.loss_fct = CrossEntropyLoss(reduction= 'none')

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tgt_pinyin_ids=None,
        pinyin_labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        retriever=None,
        gamma=1,
        var=1,
        **kwargs
    ):

        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0).long() * (input_ids != 101) * (input_ids != 102)
        outputs_x = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_x = outputs_x[0]  # (B, T, H)

        outputs_x2 = self.bert2(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
        encoded_x2 = outputs_x2[0]  # (B, T, H)
        prediction_scores = self.cls2(encoded_x2)[0]  # (B, T, V)
        B, T, H = encoded_x.shape
        V = prediction_scores.shape[2]
        device = encoded_x.device

        loss = None
        if labels is not None:
            masks_lm = loss_mask.bool()  # (B, T)

            masks_retrieve = input_ids != labels  # (B, T)

            masks_both = masks_lm  # (B, T)

            masks_copy = torch.bitwise_and(masks_lm, ~masks_retrieve)  # (B, T)
            num_copy, num_retrieve = masks_copy.sum(), masks_retrieve.sum()
            num_both = masks_both.sum()
            seq_logprobs = F.log_softmax(prediction_scores, dim=2)  # (B, T, V)
            generator_ll = seq_logprobs.gather(dim=2, index=labels.unsqueeze(2))  # (B, T, 1)
            generator_nll = -generator_ll.squeeze(2)  # (B, T)

            if self.model_type == 'lm':
                loss_generator_nll = generator_nll.masked_select(masks_lm).mean()  # (N_lm,)
                loss = loss_generator_nll if loss is None else loss + loss_generator_nll

            elif self.model_type == 'gate_copy_retr':
                gate_inputs = encoded_x2
                gate_scores = self.net_gate(encoded_x2)  # (B, T, 2)
                gate_logprobs = F.log_softmax(gate_scores, dim=-1)[masks_lm]  # (N_lm, 2)
                gate_labels = masks_retrieve[masks_lm].long()  # (N_lm,)
                loss_gate = F.cross_entropy(gate_logprobs, gate_labels, weight=torch.tensor([1., 40]).to(device))
   
                loss = loss_gate if loss is None else loss + loss_gate

                if num_both:
                    copy_gate_scores = self.net_copy_gate(gate_inputs.detach())  # (B, T, 2) detach()
                    copy_gate_logprobs = F.log_softmax(copy_gate_scores, dim=-1)[masks_both].reshape(num_both, 2, 1)  # (N_both, 2, 1)
                    
                    copy_probs = torch.zeros(B, T, V).to(device)  # (B, T, V)
                    copy_probs = torch.scatter_add(copy_probs, dim=2, index=input_ids.unsqueeze(2), src=torch.ones(B, T, 1).to(device))
                    copy_logprobs = torch.log(torch.clamp(copy_probs, min=1e-30))[masks_both]  # (N_both, V)
                    copy_all_logprobs = torch.stack([copy_logprobs, seq_logprobs[masks_both]], dim=1)   # (N_both, 2, V)

                    copy_edit_logprobs = torch.logsumexp(copy_gate_logprobs + copy_all_logprobs, dim=1)  # (N_copy, V)
                    copy_ll = copy_edit_logprobs.gather(dim=1, index=labels[masks_both].unsqueeze(1))  # (N_copy, 1)
                    copy_nll = -copy_ll.squeeze(1)  # (N_copy,)
                    loss_copy_nll = copy_nll.sum() / num_both
                    loss = loss_copy_nll if loss is None else loss + loss_copy_nll

                if num_retrieve:
                    retrieve_input = encoded_x.detach()
                    retrieve_input = retrieve_input[masks_retrieve].unsqueeze(1)  # (N_retr, 1, H)
                    retrieve_input = F.normalize(retrieve_input, p=2, dim=2)
                    dic_retrieve = retriever.retrieve(retrieve_input)
                    retrieve_outputs = retriever.getRetrieveOutputs(dic_retrieve)

                    retrieve_embeds, retrieve_values = (
                        retrieve_outputs['retrieve_embeds'].squeeze(1).float().to(device),  # (N_retr, K, H)
                        retrieve_outputs['retrieve_values'].squeeze(1).to(device),  # (N_retr, K)
                    )
                    K = retrieve_values.shape[1]

                    cross_scores = torch.bmm(encoded_x[masks_retrieve].unsqueeze(1), retrieve_embeds.transpose(1, 2)).reshape(num_retrieve, K)  # (num_mask, num_k)
                    cross_probs = torch.softmax(cross_scores, dim=1)  # (num_mask, num_k)
                    
                    retrieve_probs = torch.zeros(num_retrieve, K, V).to(device)  # (num_mask, num_k, vocab_size)
                    retrieve_probs = torch.scatter(retrieve_probs, dim=2, index=retrieve_values.unsqueeze(2), src=torch.ones(num_retrieve, K, 1).float().to(device))
                    retrieve_probs_agg = (cross_probs.unsqueeze(2) * retrieve_probs).sum(dim=1)  # (num_mask, vocab_size)
                    retrieve_logprobs_agg = torch.log(torch.clamp(retrieve_probs_agg, min=1e-30))  # (num_mask, vocab_size)
                    
                    retrieve_gate_scores = self.net_retrieve_gate(gate_inputs)  # (B, T, 2) detach()
                    retrieve_gate_logprobs = F.log_softmax(retrieve_gate_scores, dim=-1)[masks_retrieve].reshape(num_retrieve, 2, 1)  # (num_mask, 2, 1)

                    retrieve_gate_probs = torch.softmax(retrieve_gate_scores, dim=-1)[masks_retrieve].reshape(num_retrieve, 2)
                    retrieve_embeds_agg = (cross_probs.unsqueeze(2) * retrieve_embeds).sum(dim=1).to(device)  # (num_mask, dim_hidden)
                    encoded_lm = encoded_x2[masks_retrieve].reshape(num_retrieve, H)
                    encoded_all = torch.stack([encoded_lm, retrieve_embeds_agg], dim=1)  # (num_mask, 2, dim_hidden)
                    encoded_agg = (retrieve_gate_probs.unsqueeze(2) * encoded_all).sum(dim=1)
                    retrieve_prediction_scores = self.cls2(encoded_agg)[0]
                    retrieve_edit_logprobs = F.log_softmax(retrieve_prediction_scores, dim=1)
                    retrieve_ll = retrieve_edit_logprobs.gather(dim=1, index=labels[masks_retrieve].unsqueeze(1))  # (num_mask, 1)
                    retrieve_nll = -retrieve_ll.squeeze(1)  # (num_mask,)
                    loss_retrieve_nll = retrieve_nll.sum() / num_both
                    loss = loss_retrieve_nll if loss is None else loss + loss_retrieve_nll

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs_x.hidden_states,
            attentions=outputs_x.attentions,
        )
