import numpy as np
from plotly import graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sum logits and calculate probabilities and classes
def calculate_probabilities_and_classes(results):
    all_logits = []
    ankh_logits = []
    esm2_logits = []

    for model_type in results:
        for fold in results[model_type]:
            logits = results[model_type][fold].numpy()
            all_logits.append(logits)
            if model_type == "ankh":
                ankh_logits.append(logits)
            elif model_type == "esm2":
                esm2_logits.append(logits)

    all_logits_sum = np.sum(all_logits, axis=0)
    ankh_logits_sum = np.sum(ankh_logits, axis=0)
    esm2_logits_sum = np.sum(esm2_logits, axis=0)

    all_probs = torch.sigmoid(torch.tensor(all_logits_sum)).numpy()
    ankh_probs = torch.sigmoid(torch.tensor(ankh_logits_sum)).numpy()
    esm2_probs = torch.sigmoid(torch.tensor(esm2_logits_sum)).numpy()

    return all_probs, ankh_probs, esm2_probs

# Function to plot probabilities
def plot_probabilities(sequence_labels, meta_probs, esm2_probs, ankh_probs, fold_probs_esm2, fold_probs_ankh):
    fig = go.Figure()

    # Add meta average probabilities by default
    fig.add_trace(go.Scatter(
        x=sequence_labels,
        y=meta_probs,
        mode='lines',
        fill='tozeroy',
        name='Meta Average Probabilities',
        visible=True
    ))

    # Add esm2 average probabilities
    fig.add_trace(go.Scatter(
        x=sequence_labels,
        y=esm2_probs,
        mode='lines',
        fill='tozeroy',
        name='ESM2 Average Probabilities',
        visible='legendonly'
    ))

    # Add ankh average probabilities
    fig.add_trace(go.Scatter(
        x=sequence_labels,
        y=ankh_probs,
        mode='lines',
        fill='tozeroy',
        name='Ankh Average Probabilities',
        visible='legendonly'
    ))

    # Add fold probabilities for esm2 and ankh
    for fold in range(5):
        fig.add_trace(go.Scatter(
            x=sequence_labels,
            y=fold_probs_esm2[fold],
            mode='lines',
            fill='tozeroy',
            name=f'ESM2 Fold {fold} Probabilities',
            visible='legendonly'
        ))
        fig.add_trace(go.Scatter(
            x=sequence_labels,
            y=fold_probs_ankh[fold],
            mode='lines',
            fill='tozeroy',
            name=f'Ankh Fold {fold} Probabilities',
            visible='legendonly'
        ))

    # Add threshold line
    fig.add_shape(
        type='line',
        x0=0,
        y0=0.5,
        x1=len(sequence_labels) - 1,
        y1=0.5,
        line=dict(color='Red', dash='dash'),
        name='Threshold'
    )

    # Update layout
    fig.update_layout(
        title='',
        xaxis_title='Amino Acid Position',
        yaxis_title='Probabilities',
        xaxis=dict(tickmode='linear', tickvals=list(range(0, len(sequence_labels), 10)), ticktext=sequence_labels[::10], dtick=10),
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )

    fig.show()
