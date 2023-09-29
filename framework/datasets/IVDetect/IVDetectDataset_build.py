import os
import numpy as np
from tqdm import tqdm
import utils.process as process
import pandas as pd
import torch
from torch_geometric.data import Data
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def generate_glove_file(data):
    """Generates glove files."""
    sample_big_data = {
        "pdg": process.collect_code_data(data),
        "ast": process.collect_tree_info(data)
    }
    
    for key, data in sample_big_data.items():
        with open(f'glove/{key}_word.txt', 'w') as file:
            for sentence in data:
                file.write(" ".join(sentence))
                file.write("\n")

def clean_graph(_pdg_graph, _ast_nodes):
    """Cleans graph."""
    node_list = list(set(torch.flatten(_pdg_graph).tolist()) & set(_ast_nodes))
    node_list.sort()
    node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(node_list)}
    edge_index = torch.tensor([
        [node_mapping[node.item()] for node in _pdg_graph[i] if node.item() in node_mapping] 
        for i in range(2)
    ], dtype=torch.long)
    
    return edge_index, node_mapping

def load_glove_vectors(filename):
    """Loads glove vectors."""
    glove_file_dir = os.path.join(os.getcwd(), f'glove/{filename}_vectors.txt')
    tmp_file_dir = os.path.join(os.getcwd(), f'glove/{filename}_gensim.txt')
    
    if not os.path.isfile(tmp_file_dir):
        glove2word2vec(glove_file_dir, tmp_file_dir)
    return KeyedVectors.load_word2vec_format(tmp_file_dir)

def IVD_dataset_build(glove_flag=False):
    data = pd.read_csv("data.csv")
    
    if glove_flag == True:
        generate_glove_file(data)
    
    ast_glove_vector = load_glove_vectors('ast')
    pdg_glove_vector = load_glove_vectors('pdg')
    
    pdg_graphs = process.collect_pdg(data)
    feature_functions = [
        process.generate_feature_1,
        process.generate_feature_2,
        process.generate_feature_3,
        process.generate_feature_4,
        process.generate_feature_5
    ]
    
    features = [func(data, vector, 128) for func, vector in zip(feature_functions, [pdg_glove_vector, ast_glove_vector, ast_glove_vector, pdg_glove_vector, pdg_glove_vector])]
    
    for i in tqdm(range(len(data))):
        processed_features = [fea[i] for fea in features]
        code = data.at[i, "code"]
        loc = len(code.splitlines())
        
        if 5 <= loc <= 500:
            ast_nodes = list(processed_features[1].keys())
            new_pdg_graph, mapping = clean_graph(pdg_graphs[i], ast_nodes)
            
            graph_i = Data(edge_index=new_pdg_graph)
            
            new_features = []
            for fea in processed_features:
                if isinstance(fea, dict):
                    new_fea = [torch.from_numpy(np.stack(fea.get(str(key), [np.zeros(128)]))) for key in mapping.keys()]
                else:
                    new_fea = [torch.from_numpy(np.stack(item)) for item in fea if item]
                new_features.append(new_fea)
            
            graph_i.my_data = new_features
            graph_i.y = torch.tensor([data.at[i, "bug"]], dtype=int)
            
            torch.save(graph_i, os.path.join(os.getcwd(), f"data/pyg_graph/data_{i}.pt"))
            
def main(glove_flag):
    IVD_dataset_build(glove_flag)

if __name__ == '__main__':
    main(glove_flag=True)
