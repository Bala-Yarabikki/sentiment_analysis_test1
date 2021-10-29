from tokenizer_encoder import tokenizer

encoded_data_test = tokenizer.batch_encode_plus(
    df.content.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt')


input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']


dataset_test = TensorDataset(input_ids_test,
                             attention_masks_test)

len(dataset_test)


batch_size = 4

dataloader_test = DataLoader(
    dataset_test,
    sampler=RandomSampler(dataset_test),
    batch_size=batch_size)


def evaluate(dataloader_test):

    model.eval()

    loss_val_total = 0
    predictions = []

    for batch in tqdm(dataloader_test):

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1]}

        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    loss_val_avg = loss_val_total/len(dataloader_test)

    predictions = np.concatenate(predictions, axis=0)

    return loss_val_avg, predictions

