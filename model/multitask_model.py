import torch
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import (
    BartModel, PretrainedBartModel, BartClassificationHead
)
from transformers.models.bart.configuration_bart import BartConfig


class BARTmultitask(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.model.shared.num_embeddings))
        )

        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.classification_loss_fct = torch.nn.BCEWithLogitsLoss()
        self.generation_loss_fct = torch.nn.CrossEntropyLoss()


    def forward(self,
                input_ids,
                attention_mask=None,
                classification_labels=None,
                generation_labels=None,
    ):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask)
        hidden_states = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        classification_logits = self.classification_head(sentence_representation)
        classification_loss = None

        if classification_labels is not None:
            classification_labels = classification_labels.to(classification_logits.device)
            classification_loss = self.classification_loss_fct(classification_logits, classification_labels),


        generation_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        generation_loss = None

        if generation_labels is not None:
            generation_labels = generation_labels.to(generation_logits.device)
            generation_loss = self.geneariton_loss_fct(generation_logits.view(-1, self.config.vocab_size), generation_labels.view(-1))

        return {
            'classification_loss': classification_loss,
            'classification_logits': classification_logits,
            'generation_loss': generation_loss,
            'generation_logits': generation_logits,
        }



if __name__ == "__main__":
    model = BARTmultitask.from_pretrained('hyunwoongko/kobart')
    print(model)

