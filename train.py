def train(model, iterator, optimizer, head_loss_function, dep_loss_function, device):
    model.train()
    total_loss = 0

    for batch in iterator:
        # Ensure batch is a tuple of tensors
        words, pos, heads, deprels = batch
        words = words.to(device).squeeze(1)  # Adjust dimensions if necessary
        pos = pos.to(device).squeeze(1)
        heads = heads.to(device).squeeze(1)
        deprels = deprels.to(device).squeeze(1)

        optimizer.zero_grad()

        head_logits, dep_logits = model(words, pos)

        # Assuming head_logits and dep_logits are correctly shaped
        head_loss = head_loss_function(head_logits.squeeze(), heads.float())  # Adjust loss computation as needed
        dep_loss = dep_loss_function(dep_logits.view(-1, dep_vocab_size), deprels.view(-1))

        loss = head_loss + dep_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(iterator)
# Run Training
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, head_loss_function, dep_loss_function, device)
    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}')