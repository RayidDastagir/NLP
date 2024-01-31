import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from ner_functions import prepare_training_data, train_ner_model

nlp = spacy.blank("en")

    # Get the training data
train_data = prepare_training_data()

    # Train the NER model
train_ner_model(nlp, train_data)

    # Save the trained model to disk
nlp.to_disk("trained_ner_model")

# Test the trained NER model on your test text
test_text = "Subject: RATE REQUEST EX DEL TO MCT\nDear Sir,\nPlease quote your most competitive rates for below:\nEx works from Pick up C-7, Block B, Sector 14, Noida, Uttar Pradesh 201301, India to MCT, Oman\nNo of packages are –1\n1 cubic mtr\nDimensions: L 1x W 1x H 1 cm\nTotal Weight: Approx. 20kg\nCommodity - Pipes\nThanks & Regards\nDivyanshu Taxai"
doc = nlp(test_text)

# Print the extracted entities
for ent in doc.ents:
    print(f"Label: {ent.label_}, Entity: {ent.text}")