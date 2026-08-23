from pymilvus import DataType, MilvusClient
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles


MILVUS_COLLECTION = "test_from_python"


def upload_vectors(vectors) -> None:
	client = MilvusClient(
		uri="http://mini.local:20112"
	)

	if not client.has_collection(MILVUS_COLLECTION):
		schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
		schema.add_field(
			field_name="id",
			datatype=DataType.INT64,
			is_primary=True,
		)
		schema.add_field(
			field_name="vector",
			datatype=DataType.FLOAT_VECTOR,
			dim=vectors.shape[1],
		)
		client.create_collection(
			collection_name=MILVUS_COLLECTION,
			schema=schema,
		)

	client.insert(
		collection_name=MILVUS_COLLECTION,
		data=[{"vector": vector.tolist()} for vector in vectors],
	)

# Generate non-linear data
X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05)

# Fit Kernel PCA with RBF (Gaussian) kernel
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
print(X_kpca)

# Reconstruct original features if needed
X_back = kpca.inverse_transform(X_kpca)
print(X_back)

upload_vectors(X_kpca)

import matplotlib.pyplot as plt
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6, edgecolors='none')

plt.show()

