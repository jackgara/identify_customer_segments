import pandas as pd 

## Function that maps for all component : its 'weight' ? (got from cluster_centers) to features 'weigths' for that component
def cluster_components_info(kmeans, cluster, df, pca, feat_info, num_components=12, num_feat_by_component=3):
    
    # get weights of PC
    # cluster_centers_ has a shape = [PC/components, features] 
    # ie cluster_centers_[1][3] gives the 'weight' of PC 1 and feature 3
    # which means 
    weights = kmeans.cluster_centers_[cluster]
    # set component numbers
    components = list(range(len(weights)))
    # group both
    cluster_info = pd.DataFrame({"Weights":weights, "Component":components})
    #sort by more weight
    cluster_info.sort_values("Weights", ascending=False, inplace=True)

    components = []
    weights = []
    comps_info = []
    
    for index, row in cluster_info.head(n=num_components).iterrows():
        
        # get and plot weights for that component
        comp_info = plot_weights(df, pca, feat_info, 
                                                int(row["Component"]), num_feat_by_component)
        # append to list
        comps_info.append(comp_info)
        # 
        components += [int(row["Component"])] * len(comp_info)
        # weights +=  [row["Weights"]] * len(comp_info)
        
    component_info = pd.concat(comps_info, ignore_index=True)    
    # component_info.insert(0, "ComponentWeight", pd.Series(weights))
    component_info.insert(0, "Component", pd.Series(components))
        
    return component_info

def plot_weights(df, pca, feat_info, component, n_weights = 3, figsize=(7,4)):
    """
    This function displays interesting features of the selected Component 
    """
    # features 
    features = df.columns.values
    # components/eigenvectors
    components = pca.components_
    
    # get the features weights for the selected component 
    weights = dict(zip(features, components[component]))
    # sort descending, heavier first
    sorted_weights = sorted(weights.items(), key = lambda v: v[1], reverse=True)
    
    feat_names = []
    feat_weights = []
    feat_level = []
    feat_descs = []

    # get the firsts heavier n_weights
    for feat, weight in sorted_weights[:n_weights]:
        feat_names.append(feat)
        feat_weights.append(weight)
    # get the lasts heavier (negative) n_weights
    for feat, weight, in sorted_weights[-n_weights:]:
        feat_names.append(feat)
        feat_weights.append(weight)

    # add info level and description for analysis
    for feat in feat_names:
        if feat in feat_info.attribute.values:
            feat_level.append(feat_info[feat_info.attribute == feat].information_level.values[0])
            feat_descs.append(feat_info[feat_info.attribute == feat].description.values[0])
        else:
            feat_descs.append("Missing Attribute")
            feat_level.append("Missing Attribute")
    

    comp_info = {"Feature":feat_names, "Weight":feat_weights,  "Information Level":feat_level, "Description": feat_descs }
    comp_info = pd.DataFrame(comp_info)
    comp_info.sort_values("Weight", inplace=True, ascending=False)
    
    # Plot 
    df =pd.DataFrame(list(zip(feat_names,feat_weights))).set_index(0)
    ax = df.plot.bar(figsize=figsize, title="PCA Feature weights - Component {}".format(component),\
                grid=True,layout=(2,4),legend=False,rot=45)
    
    ax.set_ylabel("Feature Weight")
    ax.set_xlabel("Feature Name")

    plt.show()

    return comp_info