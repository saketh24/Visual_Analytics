import os
from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from statsmodels.multivariate.pca import PCA as pca_1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as pca_2

app = Flask(__name__)
df = pd.read_csv('data.csv', low_memory=False)
data = df[['Neighborhood', 'Bldg Type', 'Roof Style', 'House Style', 'Foundation', 'Sale Condition', 'Exterior 1st'
    , 'MS Zoning', 'Overall Qual', 'Overall Cond', 'Bedroom AbvGr', 'TotRms AbvGrd', 'Garage Area', 'SalePrice'
    , 'Lot Area', 'Gr Liv Area']]
categorical_features = ['Neighborhood', 'Bldg Type', 'Roof Style', 'House Style', 'Foundation', 'Sale Condition',
                        'Exterior 1st', 'MS Zoning']
data['Garage Area'] = data['Garage Area'].fillna(value=data['Garage Area'].mean())
data_new = data.apply(LabelEncoder().fit_transform)
for f in categorical_features:
    data[f] = data[f].astype('category')
random_sampled_data = pd.DataFrame
strat_sampled_data = pd.DataFrame
eigen_values = []
eigen_vectors = []
top_3_attr = []

def random_sampling():
    global data
    global random_sampled_data
    global categorical_features
    data_new = data.apply(LabelEncoder().fit_transform)
    random_sampled_data = data_new.sample(frac=0.25, random_state=1)


def kmeans_elbow():
    global data
    global categorical_features
    data_new = pd.get_dummies(data, columns=categorical_features)
    errors = []
    plt.figure()
    K = range(1, 10)
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(data_new)
        errors.append(model.inertia_)
    plt.plot(K, errors, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Distortion Error')
    plt.title('Elbow Method to Find Optimal K (K-Means)')
    plt.show()


def stratified_sampling():
    global data
    global categorical_features
    global strat_sampled_data
    data_new = data.apply(LabelEncoder().fit_transform)
    model = KMeans(n_clusters=3)
    model.fit(data_new)
    data_new['cluster'] = model.labels_
    s1 = data_new[data_new['cluster'] == 1]
    s2 = data_new[data_new['cluster'] == 2]
    s3 = data_new[data_new['cluster'] == 3]
    s1 = s1.sample(frac=0.25, replace=True, random_state=1)
    s2 = s2.sample(frac=0.25, replace=True, random_state=1)
    s3 = s3.sample(frac=0.25, replace=True, random_state=1)
    strat_sampled_data = pd.concat([s1, s2, s3])
    return 1


def get_eigen_values(dat):
    x = StandardScaler().fit_transform(dat)
    pca = pca_1(x, method='eig')
    temp = pca.eigenvals
    eig_values = [a/np.sum(temp) for a in temp]
    eig_vectors = pca.eigenvecs
    pca = pca_2()
    pca.fit_transform(x)
    explained_variance = list(np.cumsum(pca.explained_variance_ratio_))
    # x = np.array(dat)
    # x = x - np.mean(x, axis=0)
    # cov_mat = np.cov(x.T)
    # eig_values, eig_vectors = np.linalg.eig(cov_mat)
    # idx = eig_values.argsort()[::-1]
    # eig_values = eig_values[idx]
    # eig_vectors = eig_vectors[:, idx]
    return eig_values, eig_vectors, explained_variance


def get_top_loadings(data):
    global top_3_attr
    top_3_attr = []
    x = StandardScaler().fit_transform(data)
    pca = pca_2()
    pca.fit_transform(x)
    loadings = pca.components_.T
    sq_loadings = []
    for row in loadings:
        sq_sum = 0
        for val in row:
            sq_sum += val*val
        sq_loadings.append(sq_sum)
    top_3 = np.array(sq_loadings).argsort()[::-1][:3]
    columns = list(random_sampled_data)
    top_3_attr.append(columns[top_3[0]])
    top_3_attr.append(columns[top_3[1]])
    top_3_attr.append(columns[top_3[2]])
    return 1


@app.route('/scree_random', methods=['GET', 'POST'])
def scree_random():
    global random_sampled_data
    global eigen_values
    global eigen_vectors
    [eigen_values, eigen_vectors, explained_var] = get_eigen_values(random_sampled_data)
    labels = [i for i in range(1, len(eigen_values)+1, 1)]
    return jsonify(json.dumps({'chart_data': {'eigen_values': eigen_values, 'variance': explained_var, 'labels': labels}}))


@app.route('/scree_stratified', methods=['GET', 'POST'])
def scree_stratified():
    global strat_sampled_data
    global eigen_values
    global eigen_vectors
    [eigen_values, eigen_vectors, explained_var] = get_eigen_values(strat_sampled_data)
    labels = [i for i in range(1, len(eigen_values)+1, 1)]
    return jsonify(json.dumps({'chart_data': {'eigen_values': eigen_values, 'variance': explained_var, 'labels': labels}}))


@app.route('/scree_original', methods=['GET', 'POST'])
def scree_original():
    global data_new
    global eigen_values
    global eigen_vectors
    [eigen_values, eigen_vectors, explained_var] = get_eigen_values(data_new)
    labels = [i for i in range(1, len(eigen_values) + 1, 1)]
    return jsonify(json.dumps({'chart_data': {'eigen_values': eigen_values, 'variance': explained_var, 'labels': labels}}))


@app.route('/pca_random_data', methods = ['GET','POST'])
def pca_random_scatter():
    global random_sampled_data
    pca = pca_2(n_components=2)
    print(random_sampled_data)
    r_data = random_sampled_data
    pca_res = pca.fit_transform(r_data)
    res_data = pd.DataFrame(pca_res)
    chart_data = res_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/pca_stratified_data', methods = ['GET','POST'])
def pca_stratified_scatter():
    global strat_sampled_data
    pca = pca_2(n_components=2)
    r_data = strat_sampled_data
    pca_res = pca.fit_transform(r_data)
    res_data = pd.DataFrame(pca_res)
    chart_data = res_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/pca_original_data', methods = ['GET','POST'])
def pca_original_data():
    global data_new
    pca = pca_2(n_components=2)
    r_data = data_new
    pca_res = pca.fit_transform(r_data)
    res_data = pd.DataFrame(pca_res)
    chart_data = res_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_euclid_random_data', methods = ['GET','POST'])
def mds_euclid_random_data():
    global random_sampled_data
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    sim = pairwise_distances(random_sampled_data, metric='euclidean')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_euclid_strat_data', methods = ['GET','POST'])
def mds_euclid_strat_data():
    global strat_sampled_data
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    sim = pairwise_distances(strat_sampled_data, metric='euclidean')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_euclid_orig_data', methods = ['GET','POST'])
def mds_euclid_original_data():
    global data_new
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    sim = pairwise_distances(data_new, metric='euclidean')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_corr_random_data', methods = ['GET','POST'])
def mds_corr_random_data():
    global random_sampled_data
    mds_data = manifold.MDS(n_components=2, dissimilarity= 'precomputed')
    sim = pairwise_distances(random_sampled_data, metric='correlation')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_corr_strat_data', methods = ['GET','POST'])
def mds_corr_strat_data():
    global strat_sampled_data
    mds_data = manifold.MDS(n_components=2, dissimilarity= 'precomputed')
    sim = pairwise_distances(strat_sampled_data, metric='correlation')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/mds_corr_original_data', methods = ['GET','POST'])
def mds_corr_original_data():
    global data_new
    mds_data = manifold.MDS(n_components=2, dissimilarity= 'precomputed')
    sim = pairwise_distances(data_new, metric='correlation')
    res_data = mds_data.fit_transform(sim)
    final_data = pd.DataFrame(res_data)
    chart_data = final_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    return jsonify({'chart_data': chart_data})


@app.route('/matrix_plot_random', methods=['GET', 'POST'])
def matrix_plot_random():
    global random_sampled_data
    global top_3_attr
    get_top_loadings(random_sampled_data)
    matrix_data = pd.DataFrame()
    matrix_data[top_3_attr[0]] = random_sampled_data[top_3_attr[0]]
    matrix_data[top_3_attr[1]] = random_sampled_data[top_3_attr[1]]
    matrix_data[top_3_attr[2]] = random_sampled_data[top_3_attr[2]]
    #final_data = matrix_data.to_dict(orient='records')
    chart_data = json.dumps(matrix_data.to_dict())
    return jsonify({'chart_data': chart_data})


@app.route('/matrix_plot_strat', methods=['GET', 'POST'])
def matrix_plot_strat():
    global strat_sampled_data
    global top_3_attr
    get_top_loadings(strat_sampled_data)
    print(top_3_attr)
    matrix_data = pd.DataFrame()
    matrix_data[top_3_attr[0]] = strat_sampled_data[top_3_attr[0]]
    matrix_data[top_3_attr[1]] = strat_sampled_data[top_3_attr[1]]
    matrix_data[top_3_attr[2]] = strat_sampled_data[top_3_attr[2]]
    #final_data = matrix_data.to_dict(orient='records')
    chart_data = json.dumps(matrix_data.to_dict())
    return jsonify({'chart_data': chart_data})


@app.route('/matrix_plot_orig', methods=['GET', 'POST'])
def matrix_plot_orig():
    global data_new
    global top_3_attr
    get_top_loadings(data_new)
    matrix_data = pd.DataFrame()
    matrix_data[top_3_attr[0]] = data_new[top_3_attr[0]]
    matrix_data[top_3_attr[1]] = data_new[top_3_attr[1]]
    matrix_data[top_3_attr[2]] = data_new[top_3_attr[2]]
    #final_data = matrix_data.to_dict(orient='records')
    chart_data = json.dumps(matrix_data.to_dict())
    return jsonify({'chart_data': chart_data})

@app.route('/', methods=['GET', 'POST'])
def index():
    r1 = stratified_sampling()
    r2 = random_sampling()
    global data
    bob = scree_random()
    return render_template("index.html", data=data)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run('localhost', '5000')
