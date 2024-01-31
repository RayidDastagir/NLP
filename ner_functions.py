import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from annotations import annotations

def prepare_training_data():
    return annotations
    
def train_ner_model(nlp, train_data, n_iter=750):
    # Check if ner component is in the pipeline
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Extract unique entity labels from the training data
    unique_labels = set()
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            unique_labels.add(ent[2])

    # Add the unique entity labels to the NER component
    for label in unique_labels:
        ner.add_label(label)

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(train_data))
    train_set = train_data[:train_size]
    val_set = train_data[train_size:]

    # Disable other pipeline components during NER training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes):  # Only train NER
        # Initialize the optimizer and set an initial best loss value
        optimizer = nlp.begin_training()
        best_loss = float('inf')

        # Iterate through the specified number of training iterations
        for itn in range(n_iter):
            losses = {}

            # Iterate over batches of training data using minibatches
            for batch in minibatch(train_set, size=compounding(4.0, 32.0, 1.001)):
                texts, annotations = zip(*batch)
                example = []

                # Create Example objects for each training instance
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Update the NER model with the current batch
                nlp.update(
                    example,
                    drop=0.5,
                    losses=losses,
                )

            # Compute validation losses
            val_losses = {}
            for val_text, val_annotations in val_set:
                val_doc = nlp.make_doc(val_text)
                val_example = Example.from_dict(val_doc, val_annotations)
                nlp.update([val_example], drop=0.0, losses=val_losses)

            # Compute average validation loss
            avg_val_loss = val_losses['ner'] / len(val_set) if 'ner' in val_losses else 0.0

            # Print training and validation loss for each iteration
            print(f"Iteration {itn} - Train Loss: {losses['ner']:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Update the best loss and save the model if the current iteration has the lowest validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                nlp.to_disk("best_ner_model")

    print("Training completed. Best model saved.")


