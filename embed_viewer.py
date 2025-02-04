import numpy as np

# Path to the stored embeddings file
embedding_path = "/home/harish/Agentic_AI/embeddings/CurrentEssentialsofMedicine.npy"
embedding_path1 = "/home/harish/Agentic_AI/embeddings/MedicalDiagnosisandTreatmentMethodsinBasicMedicalSciences.npy"
# Load the embeddings
embeddings = np.load(embedding_path)
embedding = np.load(embedding_path1)
# Check the shape of the embeddings
print("Embeddings Shape:", embeddings.shape)

# View a few embeddings
print("First 2 Embeddings:\n", embeddings[:5])


# # Check the shape of the embeddings
# print("Embeddings Shape:", embedding.shape)

# # View a few embeddings
# print("First 2 Embeddings:\n", embedding[:10])
