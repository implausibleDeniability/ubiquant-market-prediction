from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import pandas as pd


def features_pca(dataframes: list, n_components: int=200) -> list:
    """Runs PCA on anonymized features
    Args:
        dataframes: list of pd.DataFrame; PCA trains on the first dataframe
            and transforms all the dataframes
        n_components: number of components for PCA
    Returns:
        list of pd.DataFrame, each dataframe processed by PCA
    """
    pca = PCA(n_components=n_components)
    pca.fit(dataframes[0].loc[:, "f_0":])
    transformed_dataframes = []
    for df in dataframes:
        pca_values = pca.transform(df.loc[:, "f_0":])
        pca_df = pd.DataFrame(pca_values, columns=[f'f_{i}' for i in range(n_components)], index=df.index)
        transformed_df = pd.concat([df.loc[:, :"investment_id"], pca_df], axis=1)    
        transformed_dataframes.append(transformed_df)
    return transformed_dataframes


def compute_kmeans_clusters(dataframes: list, n_clusters: int=200) -> list:
    """Runs MiniBatchKMeans on anonymized features
    Args:
        dataframes: list of pd.DataFrame; KMeans trains on the first dataframe
            and transforms all the dataframes
        n_clusters: number of clusters
    Returns:
        list of pd.DataFrame, each dataframe processed by PCA
    """
    clusterizer = MiniBatchKMeans(
        n_clusters=n_clusters, 
        max_iter=100,
        n_init=10, 
        batch_size=500000, 
        verbose=1)
    clusterizer.fit(dataframes[0].loc[:, "f_0":])
    transformed_dataframes = []
    for df in dataframes:
        cluster_distances = clusterizer.transform(df.loc[:, 'f_0':])
        cluster_distances = pd.DataFrame(cluster_distances, 
                                         columns=[f'dist_cluster_{i}' for i in range(n_clusters)],
                                         index=df.index
                                        )
        transformed_df = pd.concat([df, cluster_distances], axis=1)  
        transformed_dataframes.append(transformed_df)
    return transformed_dataframes


def zero_to_bool(data: pd.DataFrame) -> pd.DataFrame:
    new_columns = []
    for i in tqdm(range(300)):
        feature = data[f'f_{i}'].copy()
        if sum(abs(feature) < 1e-6) > 500:
            new_columns.append(abs(feature) < 1e-6)
            new_columns[-1].name += '_is_0'
    data_zeros = pd.concat(new_columns, axis=1)
    data = pd.concat([data, data_zeros], axis=1)
    return data