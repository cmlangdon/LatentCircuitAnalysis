"""
Plotting functions for latent circuit experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import seaborn as sns


def plot_psychometric(trial_table):
    """
    Plot psychometric heatmaps for each context for a given model.
    """

    z = trial_table.fetch('output')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')

    
    id_df = pd.DataFrame([])
    for i in range(900):
            data = {'motion_coh': motion_coh[i],
                    'color_coh': color_coh[i],
                    'z': float(np.sign(z[i][-1,1] - z[i][-1,0])),
                    'context': context[i]}

            data = pd.DataFrame(data.items())
            data = data.transpose()
            data.columns = data.iloc[0]
            data = data.drop(data.index[[0]])
            id_df = id_df.append(data)
    motion_df = id_df[id_df['context']=="motion"]
    color_df = id_df[id_df['context']=="color"]
    
    motion_df = motion_df.groupby(['motion_coh', 'color_coh'])
    color_df = color_df.groupby(['motion_coh', 'color_coh'])
    motion_df = ((motion_df['score'].apply(np.mean))).reset_index()
    color_df = ((color_df['score'].apply(np.mean))).reset_index()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    sns.heatmap(motion_df.pivot(index='motion_coh', columns='color_coh', values='score'), annot=True,vmin=0.5, vmax=1)
    plt.title('Motion context')
    
    plt.subplot(1,2,2)
    sns.heatmap(motion_df.pivot(index='motion_coh', columns='color_coh', values='score'), annot=True,vmin=0.5, vmax=1)
    plt.title('Color context')
        
def plot_mse(trial_table):
    """
    Plot mse heatmaps for each context for a given model.
    """
    plt.figure(figsize=(12, 4))
    contexts = ['motion', 'color']
    for i in range(2):
        plt.subplot(1,2,i+1)
        df = pd.DataFrame((trial_table & { 'context': contexts[i]}).fetch())
        df = df.groupby(['motion_coh', 'color_coh'])
        df = ((df['r2score'].apply(np.mean))).reset_index()
        sns.heatmap(df.pivot(index='motion_coh', columns='color_coh', values='score'), annot=True,vmax=.05)
        plt.title(contexts[i]+ ' context')

def plot_model_parameters(model_table, model_id):
    """
    Plot connectivity matrices for a model.
    """

    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1],height_ratios=[5,1]) 
    #plt.figure(figsize=(9, 4))
    # Recurrent matrix
    plt.subplot(gs[0])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_rec'), center=0, xticklabels=False, cmap='coolwarm')
    plt.title('Recurrent layer')
    
    # Input matrix
    plt.subplot(gs[1])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_in'), center=0, yticklabels=False, cmap='coolwarm')
    plt.title('Input layer')
    
    # Output matrix
    plt.subplot(gs[2])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_out'), center=0, cmap='coolwarm')
    plt.title('Output layer')

    

def plot_latent_parameters(lca_table, lca_id):
    """
    Plot connectivity and Q matrices for a given latent circuit.
    """
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1],height_ratios=[5,1], wspace=.25) 
    plt.figure(figsize=(12, 4))
    plt.subplot(gs[0])
    sns.heatmap((lca_table & {'lca_id':  lca_id}).fetch1('w_rec'), center=0, xticklabels=False, cmap='coolwarm')
    plt.title('Recurrent layer')
    
    # Input matrix
    plt.subplot(gs[1])
    sns.heatmap((lca_table & {'lca_id':  lca_id}).fetch1('w_in'), center=0, yticklabels=False, cmap='coolwarm')
    plt.title('Input layer')
    
    # Output matrix
    plt.subplot(gs[3])
    sns.heatmap((lca_table & {'lca_id':  lca_id}).fetch1('w_out'), center=0, cmap='coolwarm')
    plt.title('Output layer')


    
    # Q matrix
    plt.subplot(gs[2])
    sns.heatmap((lca_table & {'lca_id':  lca_id}).fetch1('q'), center=0, yticklabels=False, cmap='coolwarm')
    plt.title(r'$Q$')
    
  

def plot_change_of_basis(model_table, lca_table, lca_id):
    """
    Plot model connectivity in the basis Q
    """
    #model_id = (lca_table & {'lca_id':  lca_id}).fetch1('model_id')
    w_rec = (lca_table & {'lca_id':  lca_id}).fetch1('w_rec')
    w_in = (lca_table & {'lca_id':  lca_id}).fetch1('w_in')
    q = (lca_table & {'lca_id':  lca_id}).fetch1('q')
    
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1], wspace=.25) 
    plt.figure(figsize=(12, 3))
    # Recurrent matrix
    plt.subplot(gs[0])
    sns.heatmap(q.T @ w_rec @ q, center=0, xticklabels=False, cmap='coolwarm')
    plt.title(r'$Q^T W_{rec} Q$')
    
    # Input matrix
    plt.subplot(gs[1])
    sns.heatmap(q.T @ w_in, center=0, yticklabels=False, cmap='coolwarm')
    plt.title(r'$Q^T W_{in}$')
    
    

def plot_prediction_actual(model_table,lca_table, lca_trial_table,model_trial_table, lca_id, model_id):
    """
    Plot prediction v. actual scatter plot for a given latent circuit.
    """
    q = (lca_table & {'lca_id': lca_id}).fetch1('q')
    actual=(np.stack((model_trial_table & {'model_id': model_id}).fetch('hidden')) @ q.T).reshape(-1,8)
    predictions=np.stack((lca_trial_table & {'model_id': model_id, 'lca_id': lca_id}).fetch('y_pred'))[:,:,:-2].reshape(-1,8)
   
    #actual = actual @ q.T
    #predictions = predictions @ q.T
    print(predictions.shape)
    print(actual.shape)
    fig = plt.figure(figsize=(24, 8)) 
    gs = gridspec.GridSpec(2, 4, width_ratios=[ 1, 1, 1, 1]) 
    for i in range(8):
        ax = plt.subplot(gs[i])
        plt.title('Neuron '+ str(i))
        plt.scatter(predictions[:,i],actual[:,i], alpha=.15,s=5)
        plt.ylabel('Actual')
        plt.xlabel('Prediction')
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        

def plot_time_series(trial_table):
    """
    Plot time series for context, motion, color, and output axes
    """
    
    n_trials = trial_table.fetch().shape[0]
    z = trial_table.fetch('output')
    correct_choice = trial_table.fetch('correct_choice')
    context = trial_table.fetch('context')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')
    x = trial_table.fetch('hidden')

    id_df = pd.DataFrame([])
    T = 15
    for i in range(900):
        for t in range(15):
            if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
                data = {'time': t*5 ,
                        'correct_choice': correct_choice[i],
                        'motion_coh': motion_coh[i],
                        'color_coh': color_coh[i],
                        'z': float(z[i][t*5,1] - z[i][t*5,0]),
                        'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
                        'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
                        'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
                       'trial':i,
                       'context': context[i]}

                data = pd.DataFrame(data.items())
                data = data.transpose()
                data.columns = data.iloc[0]
                data = data.drop(data.index[[0]])
                id_df = id_df.append(data)

    id_df["correct_choice"] = id_df["correct_choice"].astype(float)
    id_df["z"] = id_df["z"].astype(float)
    id_df["time"] = id_df["time"].astype(float)
    id_df["context_x"] = id_df["context_x"].astype(float)
    id_df["motion_x"] = id_df["motion_x"].astype(float)
    id_df["color_x"] = id_df["color_x"].astype(float)
    id_df["motion_coh"] = id_df["motion_coh"].astype(float)
    id_df["color_coh"] = id_df["color_coh"].astype(float)
    
    plt.figure(figsize=(24, 4))
    gs = gridspec.GridSpec(1, 4,wspace=.25) 
    
    plt.subplot(gs[0])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="context_x",
    hue="context",
    palette=palette,
    legend='brief',
    lw=2,
    ci='sd'
    )
    plt.legend(loc='upper left')
        
    plt.subplot(gs[1])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="motion_x",
    hue="motion_coh",
    style='context',
    palette=palette,
    legend=True,
    lw=2,
    ci='sd'
    )
    plt.legend(loc='upper left')
        
    plt.subplot(gs[2])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="color_x",
    hue="color_coh",
    style='context',
    palette=palette,
    legend='brief',
    lw=2,
    ci='sd'
    )
    plt.legend(loc='upper left')
    
    plt.subplot(gs[3])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="z",
    hue="correct_choice",
    style='context',
    units="trial",
    palette=palette,
    legend='brief',
    lw=2,
    estimator=None
    )
    plt.legend(loc='upper left')
    

    

def plot_output(trial_table):
    """
    Plot time series for context, motion, color, and output axes
    """

    n_trials = trial_table.fetch().shape[0]
    z = trial_table.fetch('output')
    correct_choice = trial_table.fetch('correct_choice')
    context = trial_table.fetch('context')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')
    x = trial_table.fetch('hidden')

    id_df = pd.DataFrame([])
    T = 15
    for i in range(900):
        for t in range(15):
            if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
                data = {'time': t*5 ,
                        'correct_choice': correct_choice[i],
                        'motion_coh': motion_coh[i],
                        'color_coh': color_coh[i],
                        'z': float(z[i][t*5,1] - z[i][t*5,0]),
                        'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
                        'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
                        'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
                       'trial':i,
                       'context': context[i]}

                data = pd.DataFrame(data.items())
                data = data.transpose()
                data.columns = data.iloc[0]
                data = data.drop(data.index[[0]])
                id_df = id_df.append(data)

    id_df["correct_choice"] = id_df["correct_choice"].astype(float)
    id_df["z"] = id_df["z"].astype(float)
    id_df["time"] = id_df["time"].astype(float)
    id_df["context_x"] = id_df["context_x"].astype(float)
    id_df["motion_x"] = id_df["motion_x"].astype(float)
    id_df["color_x"] = id_df["color_x"].astype(float)
    id_df["motion_coh"] = id_df["motion_coh"].astype(float)
    id_df["color_coh"] = id_df["color_coh"].astype(float)

    plt.figure(figsize=(6, 4))
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="z",
    hue="correct_choice",
    style='context',
    units="trial",
    palette=palette,
    legend='brief',
    lw=2,
    estimator=None
    )
    plt.legend(loc='upper left')

    
    
def plot_correlation(lca_table, model_table):
    """
    Plot correlation between Q and W_ext
    """
    model_id = lca_table.fetch1('model_id')
    w_in = (model_table & {'model_id': model_id}).fetch1('w_in')
    w_out = (model_table & {'model_id': model_id}).fetch1('w_out')
    w_ext = np.concatenate((w_in, w_out.T), axis=1)
    q = lca_table.fetch1('q')
    
    w_ext = w_ext / np.linalg.norm(w_ext,axis=0)
    q = q / np.linalg.norm(q,axis=0)
    sns.heatmap(w_ext.T @ q)
    
    
def plot_trajectory_fit(model_trials,lca_trials,lca_table, trial):
    """
    Plot projection of a single trial onto subspace spanned by Q.
    """
    model_id = lca_table.fetch1('model_id')
    lca_id = lca_table.fetch1('lca_id')
    y_true = (model_trials & {'model_id': model_id}).fetch('hidden')[trial]
    y_pred = (lca_trials & {'lca_id': lca_id}).fetch('y_pred')[trial]
    q = lca_table.fetch1('q')
    
    y_true = y_true @ q.T
    y_pred = y_pred @ q.T
    
    fig = plt.figure(figsize=(20, 8)) 
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1]) 
    for i in range(8):
        ax = plt.subplot(gs[i])
        plt.title('Neuron '+ str(i))
        plt.plot(y_pred[:,i], label='prediction')
        plt.plot(y_true[:,i], label='actual')
        plt.legend()
        
        
        
        
        
        
        
        
        
        
        
# FIGURE TWO
def plot_figure_two(lca_table, model_id):
    """
    Accuracy vs. valid loss plot
    """
    # Make table with columns w_error, q_error, r^2
   
    df = pd.DataFrame([])
    lca_data = lca_table.fetch(as_dict=True)
    w = (Model() & {'model_id':model_id})
    q_true = (EmbeddedTrial() & {'model_id':model_id})
    for i in range(len(lca_data)):
        wbar = lca_data[i]['w_rec']
        q = lca_data[i]['q']
        
        w_error = np.linalg.norm(wbar-w) / np.linalg.norm(w)
        q_error = np.linalg.norm(q-q_true) / np.linalg.norm(q_true)
        
        data = {}

        data = pd.DataFrame(data.items())
        data = data.transpose()
        data.columns = data.iloc[0]
        data = data.drop(data.index[[0]])
        id_df = id_df.append(data)


# FIGURE THREE (Example model of context-dependent decision-making)        
def plot_figure_three(model_table, model_id,trial_table):
    
    gs = gridspec.GridSpec(2, 5, width_ratios=[5,1, 5, 5, 5],height_ratios=[5, 1],wspace=.5) 
    plt.figure(figsize=(24, 4)) 

    # Recurrent matrix
    plt.subplot(gs[0])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_rec'), center=0, xticklabels=False, cmap='coolwarm')
    plt.title('Recurrent layer')

    # Input matrix
    plt.subplot(gs[1])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_in'), center=0, yticklabels=False, cmap='coolwarm')
    plt.title('Input layer')

    # Output matrix
    plt.subplot(gs[5])
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_out'), center=0, cmap='coolwarm')
    plt.title('Output layer')

    n_trials = trial_table.fetch().shape[0]
    z = trial_table.fetch('output')
    correct_choice = trial_table.fetch('correct_choice')
    context = trial_table.fetch('context')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')
    x = trial_table.fetch('hidden')

    id_df = pd.DataFrame([])
    T = 15
    for i in range(900):
        for t in range(15):
            if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
                data = {'time': t*5 ,
                        'correct_choice': correct_choice[i],
                        'motion_coh': motion_coh[i],
                        'color_coh': color_coh[i],
                        'z': float(z[i][t*5,1] - z[i][t*5,0]),
                        'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
                        'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
                        'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
                       'trial':i,
                       'context': context[i]}

                data = pd.DataFrame(data.items())
                data = data.transpose()
                data.columns = data.iloc[0]
                data = data.drop(data.index[[0]])
                id_df = id_df.append(data)

    id_df["correct_choice"] = id_df["correct_choice"].astype(float)
    id_df["z"] = id_df["z"].astype(float)
    id_df["time"] = id_df["time"].astype(float)
    id_df["context_x"] = id_df["context_x"].astype(float)
    id_df["motion_x"] = id_df["motion_x"].astype(float)
    id_df["color_x"] = id_df["color_x"].astype(float)
    id_df["motion_coh"] = id_df["motion_coh"].astype(float)
    id_df["color_coh"] = id_df["color_coh"].astype(float)

    plt.subplot(gs[2])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="z",
    hue="correct_choice",
    style='context',
    units="trial",
    palette=palette,
    legend='brief',
    lw=2,
    estimator=None
    )
    plt.legend(loc='upper left')



    # Panel C (psychometrics)

    z = trial_table.fetch('output')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')


    id_df = pd.DataFrame([])
    for i in range(900):
            data = {'motion_coh': motion_coh[i],
                    'color_coh': color_coh[i],
                    'choice': float(np.maximum(np.sign(z[i][-1,1] - z[i][-1,0]),0)),
                    'context': context[i]}

            data = pd.DataFrame(data.items())
            data = data.transpose()
            data.columns = data.iloc[0]
            data = data.drop(data.index[[0]])
            id_df = id_df.append(data)
    motion_df = id_df[id_df['context']=="motion"]
    color_df = id_df[id_df['context']=="color"]

    motion_df = motion_df.groupby(['motion_coh', 'color_coh'])
    color_df = color_df.groupby(['motion_coh', 'color_coh'])
    motion_df = ((motion_df['choice'].apply(np.mean))).reset_index()
    color_df = ((color_df['choice'].apply(np.mean))).reset_index()


    plt.subplot(gs[3])
    sns.heatmap(motion_df.pivot(index='motion_coh', columns='color_coh', values='choice'), annot=True,vmin=0, vmax=1)
    plt.title('Motion context')

    plt.subplot(gs[4])
    sns.heatmap(color_df.pivot(index='motion_coh', columns='color_coh', values='choice'), annot=True,vmin=0, vmax=1)
    plt.title('Color context')




# FIGURE FOUR
def plot_figure_four(lca_table,trial_table):
    gs = gridspec.GridSpec(2, 4,hspace=.5) 
    plt.figure(figsize=(24, 8)) 
    
    n_trials = trial_table.fetch().shape[0]
    z = trial_table.fetch('output')
    correct_choice = trial_table.fetch('correct_choice')
    context = trial_table.fetch('context')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')
    q = lca_table.fetch1('q')
    x = np.stack(trial_table.fetch('hidden')) @ q.T

    id_df = pd.DataFrame([])
    T = 15
    for i in range(900):
        for t in range(15):
            if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
                data = {'time': t*5 ,
                        'correct_choice': correct_choice[i],
                        'motion_coh': motion_coh[i],
                        'color_coh': color_coh[i],
                        'z': float(z[i][t*5,1] - z[i][t*5,0]),
                        'context_x_motion': float(x[i][t*5,1] ),
                        'context_x_color': float(x[i][t*5,0]),
                        'motion_x_right': float(x[i][t*5,3] ),
                        'motion_x_left': float( x[i][t*5,2]),
                        'color_x_right': float(x[i][t*5,5]),
                        'color_x_left': float (x[i][t*5,4]),
                        'choice_x_right': float(x[i][t*5,7]),
                        'choice_x_left': float (x[i][t*5,6]),
                       'trial':i,
                       'context': context[i]}

                data = pd.DataFrame(data.items())
                data = data.transpose()
                data.columns = data.iloc[0]
                data = data.drop(data.index[[0]])
                id_df = id_df.append(data)

    id_df["correct_choice"] = id_df["correct_choice"].astype(float)
    id_df["z"] = id_df["z"].astype(float)
    id_df["time"] = id_df["time"].astype(float)
    id_df["context_x_motion"] = id_df["context_x_motion"].astype(float)
    id_df["context_x_color"] = id_df["context_x_color"].astype(float)
    id_df["motion_x_right"] = id_df["motion_x_right"].astype(float)
    id_df["motion_x_left"] = id_df["motion_x_left"].astype(float)
    id_df["color_x_right"] = id_df["color_x_right"].astype(float)
    id_df["choice_x_right"] = id_df["choice_x_right"].astype(float)
    id_df["choice_x_left"] = id_df["choice_x_left"].astype(float)
    id_df["color_x_left"] = id_df["color_x_left"].astype(float)
    id_df["motion_coh"] = id_df["motion_coh"].astype(float)
    id_df["color_coh"] = id_df["color_coh"].astype(float)

    # Motion context
    plt.subplot(gs[0])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="context_x_motion",
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    #Color context
    plt.subplot(gs[4])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="context_x_color",
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Motion right
    plt.subplot(gs[1])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="motion_x_right",
    hue = 'motion_coh',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Motion left
    plt.subplot(gs[5])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="motion_x_left",
    hue = 'motion_coh',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Color right
    plt.subplot(gs[2])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="color_x_right",
    hue = 'color_coh',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Color left
    plt.subplot(gs[6])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="color_x_left",
    style='context',
    hue = 'color_coh',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Choice right
    plt.subplot(gs[3])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="choice_x_right",
    hue='correct_choice',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Choice left
    plt.subplot(gs[7])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="choice_x_left",
    hue='correct_choice',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    

    
    
    
# FIGURE SIx
def plot_figure_six(lca_table, model_table):
    """
    Plot correlation between Q and W_ext
    
    """

    q = lca_table.fetch1('q')
    w_in = model_table.fetch1('w_in')
    w_out = model_table.fetch1('w_out')
    w_ext = np.concatenate((w_in,w_out.T),axis=1)
    print(q.shape)
    sns.heatmap(w_ext @ q )
    
    
    
# Figure Eight
# FIGURE FOUR
def plot_figure_eight(q, trial_table):
    gs = gridspec.GridSpec(1, 4,hspace=.5) 
    plt.figure(figsize=(24,4)) 
    
    n_trials = trial_table.fetch().shape[0]
    z = trial_table.fetch('output')
    correct_choice = trial_table.fetch('correct_choice')
    context = trial_table.fetch('context')
    motion_coh = trial_table.fetch('motion_coh')
    color_coh = trial_table.fetch('color_coh')
    x = np.stack(trial_table.fetch('hidden')) @ q

    id_df = pd.DataFrame([])
    T = 15
    for i in range(900):
        for t in range(15):
            if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
                data = {'time': t*5 ,
                        'correct_choice': correct_choice[i],
                        'motion_coh': motion_coh[i],
                        'color_coh': color_coh[i],
                        'context_x': float(x[i][t*5,0] ),
                        'motion_x': float(x[i][t*5,1] ),
                        'color_x': float(x[i][t*5,2]),
                        'choice_x': float(x[i][t*5,3]),
                       'trial':i,
                       'context': context[i]}

                data = pd.DataFrame(data.items())
                data = data.transpose()
                data.columns = data.iloc[0]
                data = data.drop(data.index[[0]])
                id_df = id_df.append(data)

    id_df["correct_choice"] = id_df["correct_choice"].astype(float)

    id_df["time"] = id_df["time"].astype(float)
    id_df["context_x"] = id_df["context_x"].astype(float)

    id_df["motion_x"] = id_df["motion_x"].astype(float)

    id_df["color_x"] = id_df["color_x"].astype(float)
    id_df["choice_x"] = id_df["choice_x"].astype(float)

    id_df["motion_coh"] = id_df["motion_coh"].astype(float)
    id_df["color_coh"] = id_df["color_coh"].astype(float)

    # context
    plt.subplot(gs[0])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="context_x",
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    #Color 
    plt.subplot(gs[1])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="color_x",
    hue='color_coh',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Motion 
    plt.subplot(gs[2])
    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=id_df,
    x="time",
    y="motion_x",
    hue = 'motion_coh',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')
    
    # Choice
    plt.subplot(gs[3])
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=id_df,
    x="time",
    y="choice_x",
    hue = 'correct_choice',
    style='context',
    palette=palette,
    legend='brief',
    )
    plt.legend(loc='upper left')

