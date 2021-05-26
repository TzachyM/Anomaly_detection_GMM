import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
from sklearn.cluster import OPTICS
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from statistics import mean
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

    #   TSNE image creator
def tsne(df, name, label=None):
    tsne = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=5000)
    tsne_data = tsne.fit_transform(df)
    plt.figure(name)
    plt.title(f'TSNE image {name}')
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=label)

    #   Preprocessing of the tabular voting data
def preprocess(df):
    column_dict = {df.columns[0]: "Name", df.columns[1]: "Symbol", df.columns[2]: "Ballot_box_num",
                   df.columns[3]: "BZB", df.columns[4]: "Voters", df.columns[5]: "Invalid", df.columns[6]: "Valid",
                   df.columns[7]: "EMT", df.columns[8]: "G", df.columns[9]: "D", df.columns[10]: "HI",
                   df.columns[11]: "HP", df.columns[12]: "HK", df.columns[13]: "V", df.columns[14]: "Z"
                , df.columns[15]: "ZH", df.columns[16]: "ZCH", df.columns[17]: "TV", df.columns[18]: "YK"
                , df.columns[19]: "KEN", df.columns[20]: "MHL", df.columns[21]: "MRZ", df.columns[22]: "N"
                , df.columns[23]: "NI", df.columns[24]: "NZ", df.columns[25]: "NK", df.columns[26]: "AM"
                , df.columns[27]: "P", df.columns[28]: "PO", df.columns[29]: "PZ", df.columns[30]: "PI"
                , df.columns[31]: "PCH", df.columns[32]: "PN", df.columns[33]: "PTZ", df.columns[34]: "TZ"
                , df.columns[35]: "ZF", df.columns[36]: "ZK", df.columns[37]: "K", df.columns[38]: "KN"
                , df.columns[39]: "RK", df.columns[40]: "SHAS"}
    df.rename(columns=column_dict, inplace=True)
    print(f"There are {df.isna().sum().sum()} NaN's in the dataset")
    df = df.groupby(['Name']).sum()
    df_unnormalized = df.copy()
    symbol = df.Symbol.copy()
    df.drop(['Symbol', 'Ballot_box_num', 'Voters', 'Invalid', 'BZB', 'Valid'], axis=1, inplace=True)
    df_unnormalized.drop(['Ballot_box_num', 'Voters', 'Invalid', 'BZB', 'Valid'], axis=1, inplace=True)
    tsne(df, 'after preprocessing')
    mandat = 29366  # mandat's size in 2013
    df = df.loc[:, df.sum(axis=0) > 2*mandat]   # Ignored votes that did not pass the voting threshold
    df = df.div(df.sum(axis=1), axis=0)     # Normalized data
    df_symbol = pd.concat([df, symbol], axis=1)
    return df, df_symbol, df_unnormalized

    #   Exploring for ideal K to use
def k_explore(df):
    K = range(2, 10)
    wss = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
        wss_iter = kmeans.inertia_  # WSS - Within-Cluster-Sum of Squared
        wss.append(wss_iter)
    mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
    plt.figure('Kmeans')
    plt.title('Ideal K - elbow graph')
    plt.scatter(x='Clusters', y='WSS', data=mycenters, marker="+")
    silh_scores = []
    #   Silhouette score
    for index, i in enumerate(range(2, 10)):
        labels = KMeans(n_clusters=i, random_state=0).fit(df).labels_
        silh_scores.append(metrics.silhouette_score(df, labels, metric='euclidean', sample_size=1000, random_state=0))
        print(f"Silhouette score for k(clusters) = {i} is {silh_scores[index]}")
    ideal_k = silh_scores.index(max(silh_scores))+2
    kmeans = KMeans(n_clusters=ideal_k, random_state=0).fit_predict(df)
    kmeans_label = pd.Series(kmeans)
    tsne(df, 'of Kmeans', kmeans_label)
    # Although it exhibit a smaller score compared to k=4, the elbow graph support k=5, but for the sake of order I went with k=4 and
    return ideal_k

    #   Detecting clusters using GMM model
def gmm_cluster_anomaly(df,k):
    gm = GaussianMixture(n_components=k, random_state=0).fit(df)
    gm_pred = gm.predict(df)
    scores = gm.score_samples(df)
    thresh = np.quantile(scores, .05)
    anomaly_index = np.where(scores <= thresh)
    print(f'\n The anomalous ballot boxes are: \n {df.iloc[anomaly_index].index.tolist()}')
    tsne = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=5000)
    tsne_data = tsne.fit_transform(df)
    fig, ax = plt.subplots()
    plt.title(f'TSNE image of clusters and anomalies')
    sc = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c=gm_pred)
    ax.scatter(tsne_data[anomaly_index, 0], tsne_data[anomaly_index, 1], color='r', label='Anomalies')
    legend1 = ax.legend(*sc.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    legend2 = ax.legend(loc='upper left')

    # dbscan = DBSCAN(eps=0.5, min_samples=30).fit_predict(df)
    # optics = OPTICS(min_samples=10, metric='euclidean').fit_predict(df)
    # Could not find the right hyper-parameters to have a good clustering distribution for DBSCAN and OPTICS.

    #   Preprocessing of voters attributes data
def voting_attributes_preprocess():
    df2 = pd.read_csv('bycode - 2013.csv')
    df2 = pd.concat([df2.iloc[:, 2],df2.iloc[:, 4], df2.iloc[:, 9]], axis=1)
    column_dict = {df2.columns[0]: "Symbol", df2.columns[1]: "Area", df2.columns[2]: "Religion"}
    df2.rename(columns=column_dict, inplace=True)
    return df2

    #   Demographic or geographic attributes of voters
def voting_attributes(df_unnormalized):
    df2 = voting_attributes_preprocess()
    print("The 5 top parties that got the most votes were:\n",
          df_unnormalized.iloc[:, 1:].sum(axis=0).sort_values(ascending=False).head(5))

    df_ = df2.merge(df_symbol, left_on='Symbol', right_on='Symbol', how='inner')
    most_votes = df_.iloc[:, 3:15].idxmax(axis=1)
    print("\nThe number of settlements where each party got a majority in\n", most_votes.value_counts().head(5))
    #   The fact that Emet (EMT) and Meretz(MRZ) parties got the majority of votes in more settlements, but still got less votes in total,
    #   can teach us that they should focus on bigger cities and not small communities. Also Shas wasnt never got a majority but had enough votes
    #   to be one of the biggest parties.
    most_votes_per_settlement = pd.concat([df_.iloc[:, 1:3], most_votes], axis=1)
    most_votes_per_settlement.rename(columns={most_votes_per_settlement.columns[2]: "Most voted party"}, inplace=True)
    print(f"There are {most_votes_per_settlement.isna().sum().sum()} NaN's in the dataset")
    most_votes_per_settlement.fillna(2, inplace=True)

    plt.figure('Area of vote: ')
    ax1 = sns.violinplot(data=most_votes_per_settlement, x=most_votes_per_settlement.iloc[:, 2],
                         y=most_votes_per_settlement.iloc[:, 0]).set_title(
        "1 - North area ; 2 - Haifa area ; 3 - Tel Aviv area ; 4 - Center area ; 5 - Jerusalem area ; 6 - South area ; 7 - Shomron area")
    plt.ylim(0, 8)
    plt.figure('Religion of voters')
    ax2 = sns.violinplot(data=most_votes_per_settlement, x=most_votes_per_settlement.iloc[:, 2],
                         y=most_votes_per_settlement.iloc[:, 1]).set_title(
        'Religion of voters: 1 - Jewish ; 2 - Arab ; 3-5 - All the rest')
    plt.ylim(0, 6)
    plt.show()
    return most_votes_per_settlement

    #   Model for prediction of the leading party in a settlement according to the religion and area in Israel
def party_majority_pred_model(most_votes_per_settlement):
    party_dict = {'EMT': 0, 'MHL': 1, 'PO': 2, 'TV': 3, 'MRZ': 4, 'AM': 5}
    most_votes_per_settlement['Most voted party'] = most_votes_per_settlement['Most voted party'].map(
        party_dict).fillna(6)
    X = most_votes_per_settlement.iloc[:, 0:2].copy()
    X[X.Religion > 1] = 2
    X = pd.get_dummies(X.astype(str))
    y = most_votes_per_settlement['Most voted party']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    RF = RandomForestClassifier(max_depth=8, random_state=0)
    cv_results = cross_validate(RF, X, y, cv=3)
    sorted(cv_results.keys())
    print("The mean score of our Random Forest model is: ", mean(cv_results['test_score']))
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    target_names = ['EMT', 'MHL', 'PO', 'TV', 'MRZ', 'AM', 'Rest']
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":

    data = pd.read_csv("df.csv", sep=',')
    df, df_symbol, df_unnormalized = preprocess(data)
    k = k_explore(df)
    gmm_cluster_anomaly(df, k)
    most_votes_per_settlement = voting_attributes(df_unnormalized)
    party_majority_pred_model(most_votes_per_settlement)

    # Our model is highly inaccurate due to the the low number of features, values and the fact that we had to use the settlement symbol
    # to concat the dataframes reduced the number of values even further. If I had more time I would have engineered more
    # features that could take into accounts the votes of parties that weren't the majority in settlements, but still
    # got a lot of votes. It was fun trying to solve it but sadly I'm short on time and I hope that this show some of my abilities.
