import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

with open('agency_to_vec.pickle', 'rb') as file:
    place_w2v = pickle.load(file)

token = list(place_w2v.keys())


def kmeans_gmm():
    min_k = 4
    max_k = 8

    K = range(min_k, max_k + 1)
    sc_scores = []

    w2v = []
    for key, value in place_w2v.items():
        w2v.append(value)

    pca = PCA(n_components=2)
    w2v_2d = pca.fit_transform(w2v)

    for k in K:
        kmeans_sk = KMeans(n_clusters=k, max_iter=600, tol=1e-6).fit(w2v_2d)
        sc_score = silhouette_score(w2v, kmeans_sk.labels_, metric='euclidean')
        sc_scores.append(sc_score)

    bestK = np.argmax(sc_scores) + min_k
    kmeans = KMeans(n_clusters=bestK, algorithm="full").fit(w2v)
    for k in range(0, 5):
        Xk = w2v_2d[kmeans.labels_ == k]
        plt.scatter(Xk[:, 0], Xk[:, 1], label=k)
        plt.legend()
    plt.title('The plot of K-means and GMM clustering')
    plt.show()
    return w2v_2d, kmeans


def plot_res(w2v_2d, kmeans_gmm_res):
    t = 1
    cmap = {0: [0.1, 0.1, 1.0, t], 1: [1.0, 0.1, 0.1, t], 2: [1.0, 0.5, 0.1, t], 3: [0.5, 1.0, 0.1, t],
            4: [0.2, 0.5, 0.9, t]}
    labels = {0: 'Label 0', 1: 'Label 1', 2: 'Label 2', 3: 'Label 3', 4: 'Label 4'}

    h = .02
    x_min, x_max = w2v_2d[:, 0].min() - 1, w2v_2d[:, 0].max() + 1
    y_min, y_max = w2v_2d[:, 1].min() - 1, w2v_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans_gmm_res.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    arrayShow = np.array([[cmap[i] for i in j] for j in Z])
    patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

    plt.figure(1)
    plt.clf()
    plt.imshow(arrayShow, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.legend(handles=patches, loc=5, borderaxespad=0.)

    plt.plot(w2v_2d[:, 0], w2v_2d[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans_gmm_res.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering \n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def main():
    w2v_2d, kmeans_gmm_res = kmeans_gmm()


if __name__ == '__main__':
    main()
