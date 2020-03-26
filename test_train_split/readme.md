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
