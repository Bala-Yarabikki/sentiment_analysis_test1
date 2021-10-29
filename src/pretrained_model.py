
model = BertForSequenceClassification.from_pretrained("'bert-base-multilingual-cased'   ",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)