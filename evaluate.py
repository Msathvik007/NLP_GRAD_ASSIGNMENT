def evaluate(model, iterator, head_loss_function, dep_loss_function, device):
    model.eval()  # Set the model to evaluation mode
    total_head_loss = 0
    total_dep_loss = 0
    correct_heads = 0
    correct_labels = 0
    total_tokens = 0

    with torch.no_grad():  # No gradients needed
        for batch in iterator:
            words, pos, heads, deprels = batch
            words = words.to(device).squeeze()
            pos = pos.to(device).squeeze()
            heads = heads.to(device).squeeze()
            deprels = deprels.to(device).squeeze()

            head_logits, dep_logits = model(words, pos)

            head_loss = head_loss_function(head_logits.squeeze(), heads.float())
            dep_loss = dep_loss_function(dep_logits.view(-1, dep_logits.size(-1)), deprels)

            total_head_loss += head_loss.item()
            total_dep_loss += dep_loss.item()

            # Convert logits to predictions
            head_preds = head_logits.round().int()  # Assuming head_logits are regression outputs
            dep_preds = dep_logits.argmax(dim=1, keepdim=True).squeeze()

            # Calculate correct predictions for UAS and LAS
            correct_heads += (head_preds == heads).sum().item()
            correct_labels += ((head_preds == heads) & (dep_preds == deprels)).sum().item()
            total_tokens += words.size(0)

    uas = correct_heads / total_tokens
    las = correct_labels / total_tokens
    return total_head_loss / len(iterator), total_dep_loss / len(iterator), uas, las

valid_dataset = DependencyParsingDataset('test.gold.conll', pos_vocab, dep_vocab)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
dev_dataset = DependencyParsingDataset('dev.gold.conll', pos_vocab, dep_vocab)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

valid_head_loss, valid_dep_loss, valid_uas, valid_las = evaluate(model, valid_loader, head_loss_function, dep_loss_function, device)
print(f'Validation Head Loss: {valid_head_loss:.4f}')
print(f'Validation Dependency Loss: {valid_dep_loss:.4f}')
print(f'Validation UAS: {valid_uas:.4f}')
print(f'Validation LAS: {valid_las:.4f}')

dev_head_loss, dev_dep_loss, dev_uas, dev_las = evaluate(model, dev_loader, head_loss_function, dep_loss_function, device)
print(f'Validation Head Loss: {dev_head_loss:.4f}')
print(f'Validation Dependency Loss: {dev_dep_loss:.4f}')
print(f'Validation UAS: {dev_uas:.4f}')
print(f'Validation LAS: {dev_las:.4f}')