import argparse
from model.Model import SweetConv
import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import softmax
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(
                    prog = 'AMBROSIA',
                    description = 'Prediction of carbohydrate binding residues on protein using pre-trained pLM embeddings.')

parser.add_argument('embeddings_path', type=str, help="""h5py file containing an entry for each chain (key) with the corresponding per-residue embedding (value)""")
parser.add_argument('parameters_path', type=str, help="""Model parameters path in PyTorch state dictionnary format""")
parser.add_argument('output_path', type=str, help="""Output path in csv format""")
parser.add_argument('-window_size', type=int, default=13, help='Sliding window size used in the model. This should be an odd integer.')
parser.add_argument('-hidden_layers', type=int, nargs='+', default=[128],
help='Number of out channels used during successive convolution layers. If a single number is specified, it is applied to all layers.')
parser.add_argument('--hidden_layers_fc', type=int, default=128,
help='Number of out channels of the first fully connected layer.')
parser.add_argument('-kernel_size', type=int, nargs='+', default=[3],
help='Kernel size of the successive convolution layers. If a single number is specified, it is applied to all layers.')
parser.add_argument('--device', type=str, default='auto', 
help="Device used for computation can be one of 'cpu', 'cuda' and 'auto'. 'auto' automatically select GPU if it exists or CPU otherwise.")

args = parser.parse_args()



def get_sliding_windows(array, window_size=13):
    """ Function creating the sliding windows of an embedding 
    Input: np.array
    Array to create sliding windows from
    Output: 
    torch.tensor containing all the sliding windows of interest"""
    # Get array size
    n_residues, n_features = array.shape
    # Adding padding and a corresponding additionnal feature
    padding = np.zeros((window_size-1, n_features))
    x = np.concatenate([padding, array, padding], axis=0)
    additional_feature = np.concatenate([np.ones(window_size-1), 
                                        np.zeros(n_residues), 
                                        np.ones(window_size-1)])
    x = np.concatenate([x, additional_feature.reshape(-1, 1)], 
                        axis=1)
    # Return stacked sliding windows
    k = window_size//2
    sw_x = sliding_window_view(x, window_size, axis=0)[k:-k]
    return torch.as_tensor(np.stack(sw_x, axis=0), dtype=torch.float)

def predict_carbohydrate_binding_sites(parameters_path, embeddings_path, 
    batch_size=1024, test_mode=None, **kwargs):
    """ Function predicting the carbohydrate binding sites using our
    embeddings and deep learning
    Input:
    - parameters_path: str
    Path to the model parameters
    - embeddings_path: str
    Path to the model embeddings (in h5py format)
    Output:
    - predictions: pd.DataFrame
    DataFrame with columns: chain_id | resid (0-based) | prediction | probability  
    """

    # Loading embeddings
    embeddings = h5py.File(embeddings_path)
    in_channels = embeddings[list(embeddings.keys())[0]][:].shape[1]+1
    # Loading model in eval mode
    model = SweetConv(in_channels=in_channels, **kwargs)

    model.load_state_dict(torch.load(parameters_path))
    
    # Setting results dic  
    results_dic = {'Chain_id': [],
                   'Resid': [],
                   'Prediction': [],
                   'Probability': []}

    # Setting eval mode and nograd
    with torch.no_grad():
        model.eval()
        # Iterating over keys
        for key in tqdm(list(embeddings.keys())[:test_mode]):
            # Create sliding windows
            sliding_windows = get_sliding_windows(embeddings[key])
            sliding_windows = DataLoader(sliding_windows, batch_size=batch_size)
            cursor = 0
            for batch in sliding_windows:
                batch.to(device)
                results_dic['Chain_id'] += [key]*len(batch)
                results_dic['Resid'] += list(range(cursor, cursor+len(batch)))
                out = model(batch)
                labels = out.cpu().numpy().argmax(1)
                results_dic['Prediction'] += labels.tolist()
                probabilities = softmax(out, dim=1, dtype=torch.float).cpu().numpy()
                results_dic['Probability'] += probabilities.max(1).tolist()
    results_dic['Probability'] = results_dic['Probability'].astype(bool)
    return pd.DataFrame(results_dic)


if __name__ == '__main__':

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    results = predict_carbohydrate_binding_sites(args.parameters_path, args.embeddings_path,
    window_size=args.window_size, hidden_layers=args.hidden_layers, 
    hidden_layers_fc=args.hidden_layers_fc, kernel_size=args.kernel_size)
    results.to_csv(args.output_path, index=False)


    


            




