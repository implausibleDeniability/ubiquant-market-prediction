from sklearn.decomposition import PCA
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