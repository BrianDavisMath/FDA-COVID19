I have included a file containing the distances to each molecule's closest 200 neighbors. By thesholding these distances you can increases/decrease the sparsity of the neighbors graph.

In the case of ligands, the distances are based on the choices of the parameters 8 and 4096. We can decrease the distances by lowering these parameters, but at the expense of some molecules becoming identified with eachother (distance zero). This is happening because the representations giving rise to the distances are lossy AND discrete, a nasty combination.

The graph has self-loops, just FYI. You can remove them if you need to.
```
import joblib
import networkx as nx
import scipy.sparse

sparse_distances_file = "sparse_distance_mat_200_ligand8 4096.joblib"
similarities = joblib.load(sparse_distances_file)
mol_neighbor_graph = nx.from_scipy_sparse_matrix(similarities)
```
The graph below focuses attention on the most similar protein in our dataset (PID P0C6X7, graph_ID 1335 below) to the largest protein associated to the coronavirus (PID QHD43415.1). The target protein QHD43415.1, if included in our graph, is only adjacent to P0C6X7, as it's similarity to any other protein in the dataset is very low.

![](https://github.com/BrianDavisMath/FDA-COVID19/blob/master/test_train_split/SARS_protein_ego_graph.png?raw=true)
