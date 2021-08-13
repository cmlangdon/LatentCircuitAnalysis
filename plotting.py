"""
Plotting functions for latent circuit experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import seaborn as sns
from datajoint_tables import *
from Trials import *
from RNN import *
from Perturbation import *
import networkx as nx

# def plot_psychometric(model_id, ax1, ax2):
#     """
#     Plot psychometric heatmaps for each context for a given model.
#     """

#     t = 3000
#     dt = .2 * 200
#     n_t = int(round(t / dt))
#     trial_events = {'n_t': int(round(n_t)),
#                     'cue_on': int(round(n_t * .1)),
#                     'cue_off': int(round(n_t * .33)),
#                     'stim_on': int(round(n_t * .4)),
#                     'stim_off': int(round(n_t)),
#                     'dec_on': int(round(n_t * .75)),
#                     'dec_off': int(round(n_t))}

#     # Load inputs and labels
#     inputs, labels, mask, conditions = generate_trials(**trial_events,
#                                               n_trials=15,
#                                               alpha=0.2,
#                                               tau=200,
#                                               sigma_in=.01,
#                                               baseline=0.2,
#                                               n_coh=6)


#     n = (Model() & {'model_id':model_id}).fetch1('n')
#     size = (Model() & {'model_id':model_id}).fetch1('connectivity')
#     rnn = RNNModule(connectivity=size, mask=mask, n=n)

#     rnn.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'))
#     rnn.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'))
#     rnn.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'))

#     z_mask, z, x = rnn.forward(inputs)
#     z = z.detach().cpu().numpy()


#     data = np.concatenate((pd.DataFrame(conditions).values,z[:,-1,1,None]-z[:,-1,0,None]),1)
#     df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','output'])
  
  
#     motion_df = df[df['context']=='motion'].groupby(['motion_coh', 'color_coh'])
#     color_df = df[df['context']=='color'].groupby(['motion_coh', 'color_coh'])
    
#     motion_df = ((motion_df['output'].apply(np.mean))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')
#     color_df = ((color_df['output'].apply(np.mean))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')

    
#     sns.heatmap(motion_df, annot=True,vmin=-1.2, vmax=1.2,center=0, ax = ax1, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2))
#     plt.title('Motion context')

#     sns.heatmap(color_df, annot=True,vmin=-1.2, vmax=1.2,center=0, ax = ax2, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2))
#     plt.title('Color context')
# def plot_psychometric(model_id):
#     """
#     Plot psychometric heatmaps for each context for a given model.
#     """

# #     z = trial_table.fetch('output')
# #     motion_coh = trial_table.fetch('motion_coh')
# #     color_coh = trial_table.fetch('color_coh')


# #     id_df = pd.DataFrame([])
# #     for i in range(900):
# #             data = {'motion_coh': motion_coh[i],
# #                     'color_coh': color_coh[i],
# #                     'choice': float(np.maximum(np.sign(z[i][-1,1] - z[i][-1,0]),0)),
# #                     'context': context[i]}

# #             data = pd.DataFrame(data.items())
# #             data = data.transpose()
# #             data.columns = data.iloc[0]
# #             data = data.drop(data.index[[0]])
# #             id_df = id_df.append(data)
# #     motion_df = id_df[id_df['context']=="motion"]
# #     color_df = id_df[id_df['context']=="color"]

# #     motion_df = motion_df.groupby(['motion_coh', 'color_coh'])
# #     color_df = color_df.groupby(['motion_coh', 'color_coh'])
# #     motion_df = ((motion_df['choice'].apply(np.mean))).reset_index()
# #     color_df = ((color_df['choice'].apply(np.mean))).reset_index()

#     t = 3000
#     dt = .2 * 200
#     n_t = int(round(t / dt))
#     trial_events = {'n_t': int(round(n_t)),
#                     'cue_on': int(round(n_t * .1)),
#                     'cue_off': int(round(n_t * .33)),
#                     'stim_on': int(round(n_t * .4)),
#                     'stim_off': int(round(n_t)),
#                     'dec_on': int(round(n_t * .75)),
#                     'dec_off': int(round(n_t))}

#     # Load inputs and labels
#     inputs, labels, mask, conditions = generate_trials(**trial_events,
#                                               n_trials=15,
#                                               alpha=0.2,
#                                               tau=200,
#                                               sigma_in=.01,
#                                               baseline=0.2,
#                                               n_coh=6)


#     n = (Model() & {'model_id':model_id}).fetch1('n')
#     size = (Model() & {'model_id':model_id}).fetch1('connectivity')
#     rnn = RNNModule(connectivity=size, mask=mask, n=n)

#     rnn.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'))
#     rnn.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'))
#     rnn.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'))

#     z_mask, z, x = rnn.forward(inputs)
#     z = z.detach().cpu().numpy()


#     data = np.concatenate((pd.DataFrame(conditions).values,z[:,-1,1,None]-z[:,-1,0,None]),1)
#     df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','output'])
#     motion_df = df[df['context']=='motion'].groupby(['motion_coh', 'color_coh'])
#     color_df = df[df['context']=='color'].groupby(['motion_coh', 'color_coh'])
    
#     motion_df = ((motion_df['output'].apply(np.mean))).reset_index()
#     color_df = ((color_df['output'].apply(np.mean))).reset_index()

#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)
#     sns.heatmap(motion_df.pivot(index='motion_coh', columns='color_coh', values='output'), annot=True,vmin=-1.2, vmax=1.2,center=0)
#     plt.title('Motion context')

#     plt.subplot(1,2,2)
#     sns.heatmap(color_df.pivot(index='motion_coh', columns='color_coh', values='output'), annot=True,vmin=-1.2, vmax=1.2,center=0)
#     plt.title('Color context')
        
# def plot_mse(trial_table):
#     """
#     Plot mse heatmaps for each context for a given model.
#     """
#     plt.figure(figsize=(12, 4))
#     contexts = ['motion', 'color']
#     for i in range(2):
#         plt.subplot(1,2,i+1)
#         df = pd.DataFrame((trial_table & { 'context': contexts[i]}).fetch())
#         df = df.groupby(['motion_coh', 'color_coh'])
#         df = ((df['r2score'].apply(np.mean))).reset_index()
#         sns.heatmap(df.pivot(index='motion_coh', columns='color_coh', values='score'), annot=True,vmax=.05)
#         plt.title(contexts[i]+ ' context')




    
  

# def plot_change_of_basis(model_table, lca_table, lca_id):
#     """
#     Plot model connectivity in the basis Q
#     """
#     #model_id = (lca_table & {'lca_id':  lca_id}).fetch1('model_id')
#     w_rec = (lca_table & {'lca_id':  lca_id}).fetch1('w_rec')
#     w_in = (lca_table & {'lca_id':  lca_id}).fetch1('w_in')
#     q = (lca_table & {'lca_id':  lca_id}).fetch1('q')
    
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1], wspace=.25) 
#     plt.figure(figsize=(12, 3))
#     # Recurrent matrix
#     plt.subplot(gs[0])
#     sns.heatmap(q.T @ w_rec @ q, center=0, xticklabels=False, cmap='coolwarm')
#     plt.title(r'$Q^T W_{rec} Q$')
    
#     # Input matrix
#     plt.subplot(gs[1])
#     sns.heatmap(q.T @ w_in, center=0, yticklabels=False, cmap='coolwarm')
#     plt.title(r'$Q^T W_{in}$')
    
    

# def plot_prediction_actual(model_table,lca_table, lca_trial_table,model_trial_table, lca_id, model_id):
#     """
#     Plot prediction v. actual scatter plot for a given latent circuit.
#     """
#     q = (lca_table & {'lca_id': lca_id}).fetch1('q')
#     actual=(np.stack((model_trial_table & {'model_id': model_id}).fetch('hidden')) @ q.T).reshape(-1,8)
#     predictions=np.stack((lca_trial_table & {'model_id': model_id, 'lca_id': lca_id}).fetch('y_pred'))[:,:,:-2].reshape(-1,8)
   
#     #actual = actual @ q.T
#     #predictions = predictions @ q.T
#     print(predictions.shape)
#     print(actual.shape)
#     fig = plt.figure(figsize=(24, 8)) 
#     gs = gridspec.GridSpec(2, 4, width_ratios=[ 1, 1, 1, 1]) 
#     for i in range(8):
#         ax = plt.subplot(gs[i])
#         plt.title('Neuron '+ str(i))
#         plt.scatter(predictions[:,i],actual[:,i], alpha=.15,s=5)
#         plt.ylabel('Actual')
#         plt.xlabel('Prediction')
#         lims = [
#         np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#         np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#         ]

#         # now plot both limits against eachother
#         ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#         ax.set_aspect('equal')
#         ax.set_xlim(lims)
#         ax.set_ylim(lims)
        

# def plot_time_series(trial_table):
#     """
#     Plot time series for context, motion, color, and output axes
#     """
    
#     n_trials = trial_table.fetch().shape[0]
#     z = trial_table.fetch('output')
#     correct_choice = trial_table.fetch('correct_choice')
#     context = trial_table.fetch('context')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')
#     x = trial_table.fetch('hidden')

#     id_df = pd.DataFrame([])
#     T = 15
#     for i in range(900):
#         for t in range(15):
#             if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
#                 data = {'time': t*5 ,
#                         'correct_choice': correct_choice[i],
#                         'motion_coh': motion_coh[i],
#                         'color_coh': color_coh[i],
#                         'z': float(z[i][t*5,1] - z[i][t*5,0]),
#                         'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
#                         'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
#                         'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
#                        'trial':i,
#                        'context': context[i]}

#                 data = pd.DataFrame(data.items())
#                 data = data.transpose()
#                 data.columns = data.iloc[0]
#                 data = data.drop(data.index[[0]])
#                 id_df = id_df.append(data)

#     id_df["correct_choice"] = id_df["correct_choice"].astype(float)
#     id_df["z"] = id_df["z"].astype(float)
#     id_df["time"] = id_df["time"].astype(float)
#     id_df["context_x"] = id_df["context_x"].astype(float)
#     id_df["motion_x"] = id_df["motion_x"].astype(float)
#     id_df["color_x"] = id_df["color_x"].astype(float)
#     id_df["motion_coh"] = id_df["motion_coh"].astype(float)
#     id_df["color_coh"] = id_df["color_coh"].astype(float)
    
#     plt.figure(figsize=(24, 4))
#     gs = gridspec.GridSpec(1, 4,wspace=.25) 
    
#     plt.subplot(gs[0])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="context_x",
#     hue="context",
#     palette=palette,
#     legend='brief',
#     lw=2,
#     ci='sd'
#     )
#     plt.legend(loc='upper left')
        
#     plt.subplot(gs[1])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="motion_x",
#     hue="motion_coh",
#     style='context',
#     palette=palette,
#     legend=True,
#     lw=2,
#     ci='sd'
#     )
#     plt.legend(loc='upper left')
        
#     plt.subplot(gs[2])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="color_x",
#     hue="color_coh",
#     style='context',
#     palette=palette,
#     legend='brief',
#     lw=2,
#     ci='sd'
#     )
#     plt.legend(loc='upper left')
    
#     plt.subplot(gs[3])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="z",
#     hue="correct_choice",
#     style='context',
#     units="trial",
#     palette=palette,
#     legend='brief',
#     lw=2,
#     estimator=None
#     )
#     plt.legend(loc='upper left')
    


# FIGURES

def plot_latent_parameters(lca_table, lca_id, ax1, ax2, ax3, vmin,vmax):
    """
    Plot connectivity and Q matrices for a given latent circuit.
    """

    w_rec = (lca_table & {'lca_id':  lca_id}).fetch1('w_rec')
    w_in = (lca_table & {'lca_id':  lca_id}).fetch1('w_in')
    w_out = (lca_table & {'lca_id':  lca_id}).fetch1('w_out')
    
#   sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_rec'), center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax1,vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": 3})
    sns.heatmap(w_rec, center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax1, cbar=True, vmin=vmin,vmax=vmax)

    ax1.axhline(y=0, color='k',linewidth=1)
    ax1.axhline(y=w_rec.shape[0], color='k',linewidth=1)
    ax1.axvline(x=0, color='k',linewidth=1)
    ax1.axvline(x=w_rec.shape[0], color='k',linewidth=1)

    
    # Input matrix
   
    sns.heatmap(w_in, center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax2, cbar=False)
    ax2.axhline(y=0, color='k',linewidth=1)
    ax2.axhline(y=w_in.shape[0], color='k',linewidth=1)
    ax2.axvline(x=0, color='k',linewidth=1)
    ax2.axvline(x=w_rec.shape[1], color='k',linewidth=1)
    
    # Output matrix
   
    sns.heatmap(w_out,xticklabels=False,yticklabels=False, center=0, cmap='coolwarm', ax = ax3, cbar=False)
    ax3.axhline(y=0, color='k',linewidth=1)
    ax3.axhline(y=w_out.shape[0], color='k',linewidth=1)
    ax3.axvline(x=0, color='k',linewidth=1)
    ax3.axvline(x=w_out.shape[1], color='k',linewidth=1)



    
def plot_circuit_network(model_table, model_id, ax):
        # Panel A (stimulus space)
 
    w_rec = (Model() & {'model_id':  "LZCI7OM8"}).fetch1('w_rec')
    #w_rec = (np.abs(w_rec)>.3) * w_rec
    w_in = (Model() & {'model_id':  "LZCI7OM8"}).fetch1('w_in')
    #w_in = (np.abs(w_in)>.3) * w_in
    w_out = (Model() & {'model_id':  "LZCI7OM8"}).fetch1('w_out')

    
    A=np.zeros((8, 8))
    A[:8,:8]=w_rec
    #A[:8,10:]=w_in
    #A[8:10,:8]=w_out
    G = nx.from_numpy_matrix(A.T, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    pos = {0: [0,.5],
          1: [0,-.5],
          2: [1,2],
          3: [1,1],
           4:[1,-1],
           5: [1,-2],
           6: [2, .5],
           7: [2,-.5]
          }

    nx.draw_networkx_nodes(G, pos,node_color='lightgray',
                           edgecolors='black',
                           linewidths=.5,
                           node_size=50,
                           alpha=1,
                          ax=ax)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    nx.draw_networkx_edges(G, pos,edge_color=weights,connectionstyle='Angle3',edge_cmap=plt.cm.coolwarm,arrowsize=10, alpha = .25, ax = ax)
    
    # highlights
    A=np.zeros((8, 8))
    A[2:6,:2] = w_rec[2:6,:2]
    A[6:,2:6] = w_rec[6:,2:6]
    G = nx.from_numpy_matrix(A.T, create_using=nx.DiGraph)

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    nx.draw_networkx_edges(G, pos,edge_color=weights,connectionstyle='Angle3',edge_cmap=plt.cm.coolwarm,arrowsize=10, alpha = 1, ax = ax)
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    #ax.text(pos[10][0] + .25, pos[10][1],s='Context \m input',horizontalalignment='center', fontsize=6)
    #ax.text(pos[12][0],pos[12][1]-0.25,s='Motion \n input',horizontalalignment='center', fontsize=6)

    #cut = 1.05
    #xmax= cut*max(xx for xx,yy in pos.values())
    #ymax= cut*max(yy for xx,yy in pos.values())

    #ax.set_xlim(-xmax, xmax)
    # ax.set_ylim(-ymax, ymax)

def plot_circuit_decomposition(model_table, model_id, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
    """
    Plot connectivity matrices for a model.
    """
    fontsize = 3
    linewidth = 1
    cmap = 'coolwarm'
    w_rec = (model_table & {'model_id':  model_id}).fetch1('w_rec')
    
    sns.heatmap(w_rec[:2,:2], center=0, cmap=cmap, ax = ax1, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax1.axhline(y=0, color='k',linewidth=linewidth)
    ax1.axhline(y=2, color='k',linewidth=linewidth)
    ax1.axvline(x=0, color='k',linewidth=linewidth)
    ax1.axvline(x=2, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[:2,2:6], center=0,  cmap=cmap, ax = ax2, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax2.axhline(y=0, color='k',linewidth=linewidth)
    ax2.axhline(y=2, color='k',linewidth=linewidth)
    ax2.axvline(x=0, color='k',linewidth=linewidth)
    ax2.axvline(x=4, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[:2,6:], center=0,  cmap=cmap, ax = ax3, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax3.axhline(y=0, color='k',linewidth=linewidth)
    ax3.axhline(y=2, color='k',linewidth=linewidth)
    ax3.axvline(x=0, color='k',linewidth=linewidth)
    ax3.axvline(x=2, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[2:6,:2], center=0,  cmap=cmap, ax = ax4, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax4.axhline(y=0, color='k',linewidth=linewidth)
    ax4.axhline(y=4, color='k',linewidth=linewidth)
    ax4.axvline(x=0, color='k',linewidth=linewidth)
    ax4.axvline(x=2, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[2:6,2:6], center=0,  cmap=cmap, ax = ax5, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax5.axhline(y=0, color='k',linewidth=linewidth)
    ax5.axhline(y=4, color='k',linewidth=linewidth)
    ax5.axvline(x=0, color='k',linewidth=linewidth)
    ax5.axvline(x=4, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[2:6,6:], center=0,  cmap=cmap, ax = ax6, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax6.axhline(y=0, color='k',linewidth=linewidth)
    ax6.axhline(y=4, color='k',linewidth=linewidth)
    ax6.axvline(x=0, color='k',linewidth=linewidth)
    ax6.axvline(x=2, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[6:,:2], center=0,  cmap=cmap, ax = ax7, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax7.axhline(y=0, color='k',linewidth=linewidth)
    ax7.axhline(y=2, color='k',linewidth=linewidth)
    ax7.axvline(x=0, color='k',linewidth=linewidth)
    ax7.axvline(x=2, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[6:,2:6], center=0, cmap=cmap, ax = ax8, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax8.axhline(y=0, color='k',linewidth=linewidth)
    ax8.axhline(y=2, color='k',linewidth=linewidth)
    ax8.axvline(x=0, color='k',linewidth=linewidth)
    ax8.axvline(x=4, color='k',linewidth=linewidth)
    
    sns.heatmap(w_rec[6:,6:], center=0,  cmap=cmap, ax = ax9, vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": fontsize}, xticklabels=False, yticklabels=False)
    ax9.axhline(y=0, color='k',linewidth=linewidth)
    ax9.axhline(y=2, color='k',linewidth=linewidth)
    ax9.axvline(x=0, color='k',linewidth=linewidth)
    ax9.axvline(x=2, color='k',linewidth=linewidth)
    
#     sns.set(font_scale = 2)


    


def plot_model_parameters(model_table, model_id, ax1, ax2, ax3, vmin,vmax):
    """
    Plot connectivity matrices for a model.
    """


    w_rec = (model_table & {'model_id':  model_id}).fetch1('w_rec')
    w_in = (model_table & {'model_id':  model_id}).fetch1('w_in')
    w_out = (model_table & {'model_id':  model_id}).fetch1('w_out')
    
#   sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_rec'), center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax1,vmin=-1.5, vmax=1.5, cbar=False, annot=True,annot_kws={"size": 3})
    sns.heatmap(w_rec, center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax1,vmin=vmin, vmax=vmax, cbar=False)

    ax1.axhline(y=0, color='k',linewidth=1)
    ax1.axhline(y=w_rec.shape[0], color='k',linewidth=1)
    ax1.axvline(x=0, color='k',linewidth=1)
    ax1.axvline(x=w_rec.shape[0], color='k',linewidth=1)

    
    # Input matrix
   
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_in'), center=0, xticklabels=False,yticklabels=False, cmap='coolwarm', ax = ax2, cbar=False)
    ax2.axhline(y=0, color='k',linewidth=1)
    ax2.axhline(y=w_in.shape[0], color='k',linewidth=1)
    ax2.axvline(x=0, color='k',linewidth=1)
    ax2.axvline(x=w_rec.shape[1], color='k',linewidth=1)
    
    # Output matrix
   
    sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_out'),xticklabels=False,yticklabels=False, center=0, cmap='coolwarm', ax = ax3, cbar=False)
    ax3.axhline(y=0, color='k',linewidth=1)
    ax3.axhline(y=w_out.shape[0], color='k',linewidth=1)
    ax3.axvline(x=0, color='k',linewidth=1)
    ax3.axvline(x=w_out.shape[1], color='k',linewidth=1)

    

    
def plot_sorted_projections(model_id, q, ax1, ax2, ax3, ax4,ax5):
    
    t = 3000
    dt = .2 * 200
    n_t = int(round(t / dt))
    trial_events = {'n_t': int(round(n_t)),
                    'cue_on': int(round(n_t * .1)),
                    'cue_off': int(round(n_t * .33)),
                    'stim_on': int(round(n_t * .4)),
                    'stim_off': int(round(n_t)),
                    'dec_on': int(round(n_t * .75)),
                    'dec_off': int(round(n_t))}

    # Load inputs and labels
    inputs, labels, mask, conditions = generate_trials(**trial_events,
                                              n_trials=15,
                                              alpha=0.2,
                                              tau=200,
                                              sigma_in=.01,
                                              baseline=0.2,
                                              n_coh=6)


    n = (Model() & {'model_id':model_id}).fetch1('n')
    size = (Model() & {'model_id':model_id}).fetch1('connectivity')
    rnn = RNNModule(connectivity=size, mask=mask, n=n)

    rnn.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'))
    rnn.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'))
    rnn.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'))

    z_mask, z, x = rnn.forward(inputs)
    x = x[:,::5,:].detach().numpy()
    z = z[:,::5,:].detach().numpy()
    time = np.tile(40 * 5 * np.arange(x.shape[1]),x.shape[0])[:,None]
    x = np.reshape(x,(-1,x.shape[2])) @ q.T
    z = np.diff(np.reshape(z,(-1,z.shape[2])))
    
    conditions = np.repeat(pd.DataFrame(conditions).values, repeats=15, axis=0)

    data = np.concatenate((conditions,time,x,z),1)
    df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','time','context_x','motion_x','color_x', 'choice_x','z'])

    df["correct_choice"] = df["correct_choice"].astype(float)
    df["time"] = df["time"].astype(float)
    df["context_x"] = df["context_x"].astype(float)
    df["motion_x"] = df["motion_x"].astype(float)
    df["color_x"] = df["color_x"].astype(float)
    df["choice_x"] = df["choice_x"].astype(float)
    df["motion_coh"] = df["motion_coh"].astype(float)
    df["z"] = df["z"].astype(float)

     # Plotting
    fontsize = 8
  
  
    sns.lineplot(
    data=df,
    x="time",
    y="context_x",
    style='context',
    color='gray',
    legend=False,
    ax = ax1
    )
    #ax1.set(xticklabels=[])
#     for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
#         label.set_fontsize(8)
    
    #ax1.set(yticklabels=[])
    
    ax1.set_ylabel("")
    ax1.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax1.tick_params(bottom=False)
    ax1.tick_params(left=False)
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    #plt.legend(loc='upper left')

    #Motion 

    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=df,
    x="time",
    y="motion_x",
    hue='motion_coh',
    style='context',
    palette=palette,
    legend=False,
        ax = ax2
    )
    #ax2.set(xticklabels=[])
#     for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
#         label.set_fontsize(8)
   # ax2.set(yticklabels=[])
    
    ax2.set_ylabel("")
    ax2.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax2.tick_params(bottom=False)
    ax2.tick_params(left=False)
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    #plt.legend(loc='upper left')

    # Color 

    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=df,
    x="time",
    y="color_x",
    hue = 'color_coh',
    style='context',
    palette=palette,
    legend=False,
        ax = ax3
    )
    #ax3.set(xticklabels=[])
#     for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
#         label.set_fontsize(8)
    #ax3.set(yticklabels=[])
    
    ax3.set_ylabel("")
    ax3.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax3.tick_params(bottom=False)
    ax3.tick_params(left=False)
    
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    #plt.legend(loc='upper left')

    # Choice
  
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=df,
    x="time",
    y="choice_x",
    hue = 'correct_choice',
    style='context',
    palette=palette,
    legend=False,
        ax = ax4
    )
    #ax4.set(xticklabels=[])
    for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
        label.set_fontsize(8)
    #ax4.set(yticklabels=[])
    
    ax4.set_ylabel("")
    ax4.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax4.tick_params(bottom=False)
    ax4.tick_params(left=False)
    
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)


    #plt.legend(loc='upper left')
    #plt.legend(loc='upper left')
    
    
        # Output
  
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=df,
    x="time",
    y="z",
    hue = 'correct_choice',
    style='context',
    palette=palette,
    legend=False,
        ax = ax5
    )
    #ax4.set(xticklabels=[])
#     for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
#         label.set_fontsize(8)
    #ax5.set(yticklabels=[])
    
    ax5.set_ylabel("")
    ax5.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax5.tick_params(bottom=False)
    ax5.tick_params(left=False)
    
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)  
    
    
#     sns.set(font_scale = 2)
def plot_sorted_lca_projections(lca_id,q, ax1, ax2, ax3, ax4, ax5):
    
    t = 3000
    dt = .2 * 200
    n_t = int(round(t / dt))
    trial_events = {'n_t': int(round(n_t)),
                    'cue_on': int(round(n_t * .1)),
                    'cue_off': int(round(n_t * .33)),
                    'stim_on': int(round(n_t * .4)),
                    'stim_off': int(round(n_t)),
                    'dec_on': int(round(n_t * .75)),
                    'dec_off': int(round(n_t))}

    # Load inputs and labels
    inputs, labels, mask, conditions = generate_trials(**trial_events,
                                              n_trials=15,
                                              alpha=0.2,
                                              tau=200,
                                              sigma_in=.01,
                                              baseline=0.2,
                                              n_coh=6)


    n = 8

    rnn = RNNModule(connectivity='small', mask=mask, n=n)

    rnn.recurrent_layer.weight.data = torch.tensor((LCA() & {'lca_id':lca_id}).fetch1('w_rec'))
    rnn.input_layer.weight.data = torch.tensor((LCA() & {'lca_id':lca_id}).fetch1('w_in'))
    rnn.output_layer.weight.data = torch.tensor((LCA() & {'lca_id':lca_id}).fetch1('w_out'))

    z_mask, z, x = rnn.forward(inputs)
    x = x[:,::5,:].detach().numpy()
    z = z[:,::5,:].detach().numpy()
    time = np.tile(40 * 5 * np.arange(x.shape[1]),x.shape[0])[:,None]
    x = np.reshape(x,(-1,x.shape[2])) @ q.T
    z = np.diff(np.reshape(z,(-1,z.shape[2])))
    
    conditions = np.repeat(pd.DataFrame(conditions).values, repeats=15, axis=0)

    data = np.concatenate((conditions,time,x,z),1)
    df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','time','context_x','motion_x','color_x', 'choice_x','z'])

    df["correct_choice"] = df["correct_choice"].astype(float)
    df["time"] = df["time"].astype(float)
    df["context_x"] = df["context_x"].astype(float)
    df["motion_x"] = df["motion_x"].astype(float)
    df["color_x"] = df["color_x"].astype(float)
    df["choice_x"] = df["choice_x"].astype(float)
    df["motion_coh"] = df["motion_coh"].astype(float)
    df["z"] = df["z"].astype(float)


     # Plotting
    fontsize = 8
  
  
    sns.lineplot(
    data=df,
    x="time",
    y="context_x",
    style='context',
    color='gray',
    legend=False,
    ax = ax1
    )
    #ax1.set(xticklabels=[])
#     for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
#         label.set_fontsize(8)
    
    ax1.set(yticklabels=[])
    
    ax1.set_ylabel("")
    ax1.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax1.tick_params(bottom=False)
    ax1.tick_params(left=False)
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    #plt.legend(loc='upper left')

    #Motion 

    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=df,
    x="time",
    y="motion_x",
    hue='motion_coh',
    style='context',
    palette=palette,
    legend=False,
        ax = ax2
    )
    #ax2.set(xticklabels=[])
#     for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
#         label.set_fontsize(8)
    #ax2.set(yticklabels=[])
    
    ax2.set_ylabel("")
    ax2.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax2.tick_params(bottom=False)
    ax2.tick_params(left=False)
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    #plt.legend(loc='upper left')

    # Color 

    palette = sns.color_palette("coolwarm", 6)
    sns.lineplot(
    data=df,
    x="time",
    y="color_x",
    hue = 'color_coh',
    style='context',
    palette=palette,
    legend=False,
        ax = ax3
    )
    #ax3.set(xticklabels=[])
#     for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
#         label.set_fontsize(8)
    #ax3.set(yticklabels=[])
    
    ax3.set_ylabel("")
    ax3.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax3.tick_params(bottom=False)
    ax3.tick_params(left=False)
    
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    #plt.legend(loc='upper left')

    # Choice
  
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=df,
    x="time",
    y="choice_x",
    hue = 'correct_choice',
    style='context',
    palette=palette,
    legend=False,
        ax = ax4
    )
    #ax4.set(xticklabels=[])
#     for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
#         label.set_fontsize(8)
    #ax4.set(yticklabels=[])
    
    ax4.set_ylabel("")
    ax4.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax4.tick_params(bottom=False)
    ax4.tick_params(left=False)
    
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)  
    
        # Output
  
    palette = sns.color_palette("coolwarm", 2)
    sns.lineplot(
    data=df,
    x="time",
    y="z",
    hue = 'correct_choice',
    style='context',
    palette=palette,
    legend=False,
        ax = ax5
    )
#     #ax4.set(xticklabels=[])
#     for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
#         label.set_fontsize(8)
    #ax5.set(yticklabels=[])
    
    ax5.set_ylabel("")
    ax5.set_xlabel("Time (ms)",fontsize=fontsize)
    
    ax5.tick_params(bottom=False)
    ax5.tick_params(left=False)
    
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)  
    
    
def plot_perturbation(model_id, lca_id, strength, column, ax1,ax2):
    query = Model() & {'model_id': model_id}
    perturbation = Perturbation(tau=200,
                                sigma_rec=float(0.15),
                                n=query.fetch1('n'))
    perturbation.input_layer.weight.data = torch.tensor(query.fetch1('w_in')).float()
    perturbation.recurrent_layer.weight.data = torch.tensor(query.fetch1('w_rec')).float()
    perturbation.output_layer.weight.data = torch.tensor(query.fetch1('w_out')).float()
    perturbation.to(device=device)

    # Define trial structure.
    t = 3000
    dt = 200 * .2
    n_t = int(round(t / dt))
    trial_events = {'n_t': int(round(n_t)),
                    'cue_on': int(round(n_t * .1)),
                    'cue_off': int(round(n_t * .33)),
                    'stim_on': int(round(n_t * .4)),
                    'stim_off': int(round(n_t)),
                    'dec_on': int(round(n_t * .75)),
                    'dec_off': int(round(n_t))}

    # Generate dataset for simulation

    inputs, labels, mask, conditions  = generate_trials(**trial_events,
                                                n_trials=25,
                                                alpha=float(0.2),
                                                tau=perturbation.tau,
                                                sigma_in=.01,
                                                baseline=0.2,
                                                n_coh=6)

    q = (LCA() & {'lca_id': lca_id}).fetch1('q')

    
    p = q[None,column,:]
    p = strength * p / np.linalg.norm(p, axis=0)
    p = torch.tensor(p).float().to(device=device)

    inputs = inputs.to(device)

    z, x = perturbation.forward(inputs,p, float(0.2))
    z = z.detach().cpu().numpy()

    data = np.concatenate((pd.DataFrame(conditions).values,np.maximum(np.sign(z[:,-1,1,None]-z[:,-1,0,None]),0)),1)
    df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','output'])
    
    sns.lineplot(data=df[df['context']=='motion'],x='motion_coh',y='output')
  
#     motion_df = df[df['context']=='motion'].groupby(['motion_coh', 'color_coh'])
#     color_df = df[df['context']=='color'].groupby(['motion_coh', 'color_coh'])
    
#     motion_df = ((motion_df['output'].apply(np.mean ))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')
#     color_df = ((color_df['output'].apply(np.mean))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')

    
#     sns.heatmap(motion_df, annot=True,vmin=-0, vmax=1, ax = ax1, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2), cbar=False,annot_kws={"size": 8})
#     plt.title('Motion context')
#     for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
#         label.set_fontsize(8)

#     sns.heatmap(color_df, annot=True,vmin=0, vmax=1, ax = ax2, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2), cbar=False,annot_kws={"size": 8})
#     plt.title('Color context')
#     for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
#         label.set_fontsize(8)
        
    

    
def plot_model(model_id):
    model_table = Model()
    trial_table = Trial() & {'model_id': model_id}
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
    
    
    

    

# def plot_output(trial_table):
#     """
#     Plot time series for context, motion, color, and output axes
#     """

#     n_trials = trial_table.fetch().shape[0]
#     z = trial_table.fetch('output')
#     correct_choice = trial_table.fetch('correct_choice')
#     context = trial_table.fetch('context')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')
#     x = trial_table.fetch('hidden')

#     id_df = pd.DataFrame([])
#     T = 15
#     for i in range(900):
#         for t in range(15):
#             if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
#                 data = {'time': t*5 ,
#                         'correct_choice': correct_choice[i],
#                         'motion_coh': motion_coh[i],
#                         'color_coh': color_coh[i],
#                         'z': float(z[i][t*5,1] - z[i][t*5,0]),
#                         'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
#                         'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
#                         'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
#                        'trial':i,
#                        'context': context[i]}

#                 data = pd.DataFrame(data.items())
#                 data = data.transpose()
#                 data.columns = data.iloc[0]
#                 data = data.drop(data.index[[0]])
#                 id_df = id_df.append(data)

#     id_df["correct_choice"] = id_df["correct_choice"].astype(float)
#     id_df["z"] = id_df["z"].astype(float)
#     id_df["time"] = id_df["time"].astype(float)
#     id_df["context_x"] = id_df["context_x"].astype(float)
#     id_df["motion_x"] = id_df["motion_x"].astype(float)
#     id_df["color_x"] = id_df["color_x"].astype(float)
#     id_df["motion_coh"] = id_df["motion_coh"].astype(float)
#     id_df["color_coh"] = id_df["color_coh"].astype(float)

#     plt.figure(figsize=(6, 4))
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="z",
#     hue="correct_choice",
#     style='context',
#     units="trial",
#     palette=palette,
#     legend='brief',
#     lw=2,
#     estimator=None
#     )
#     plt.legend(loc='upper left')

    
    
# def plot_correlation(lca_table, model_table):
#     """
#     Plot correlation between Q and W_ext
#     """
#     model_id = lca_table.fetch1('model_id')
#     w_in = (model_table & {'model_id': model_id}).fetch1('w_in')
#     w_out = (model_table & {'model_id': model_id}).fetch1('w_out')
#     w_ext = np.concatenate((w_in, w_out.T), axis=1)
#     q = lca_table.fetch1('q')
    
#     w_ext = w_ext / np.linalg.norm(w_ext,axis=0)
#     q = q / np.linalg.norm(q,axis=0)
#     sns.heatmap(w_ext.T @ q)
    
    
# def plot_trajectory_fit(model_trials,lca_trials,lca_table, trial):
#     """
#     Plot projection of a single trial onto subspace spanned by Q.
#     """
#     model_id = lca_table.fetch1('model_id')
#     lca_id = lca_table.fetch1('lca_id')
#     y_true = (model_trials & {'model_id': model_id}).fetch('hidden')[trial]
#     y_pred = (lca_trials & {'lca_id': lca_id}).fetch('y_pred')[trial]
#     q = lca_table.fetch1('q')
    
#     y_true = y_true @ q.T
#     y_pred = y_pred @ q.T
    
#     fig = plt.figure(figsize=(20, 8)) 
#     gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1]) 
#     for i in range(8):
#         ax = plt.subplot(gs[i])
#         plt.title('Neuron '+ str(i))
#         plt.plot(y_pred[:,i], label='prediction')
#         plt.plot(y_true[:,i], label='actual')
#         plt.legend()
        
        
        
        
        
        
        
        
        
        
        
# # FIGURE TWO
# def plot_figure_two(lca_table, model_id):
#     """
#     Accuracy vs. valid loss plot
#     """
#     # Make table with columns w_error, q_error, r^2
   
#     df = pd.DataFrame([])
#     lca_data = lca_table.fetch(as_dict=True)
#     w = (Model() & {'model_id':model_id})
#     q_true = (EmbeddedTrial() & {'model_id':model_id})
#     for i in range(len(lca_data)):
#         wbar = lca_data[i]['w_rec']
#         q = lca_data[i]['q']
        
#         w_error = np.linalg.norm(wbar-w) / np.linalg.norm(w)
#         q_error = np.linalg.norm(q-q_true) / np.linalg.norm(q_true)
        
#         data = {}

#         data = pd.DataFrame(data.items())
#         data = data.transpose()
#         data.columns = data.iloc[0]
#         data = data.drop(data.index[[0]])
#         id_df = id_df.append(data)


# # FIGURE THREE (Example model of context-dependent decision-making)        
# def plot_figure_three(model_table, model_id,trial_table):
    
#     gs = gridspec.GridSpec(2, 5, width_ratios=[5,1, 5, 5, 5],height_ratios=[5, 1],wspace=.5) 
#     plt.figure(figsize=(24, 4)) 

#     # Recurrent matrix
#     plt.subplot(gs[0])
#     sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_rec'), center=0, xticklabels=False, cmap='coolwarm')
#     plt.title('Recurrent layer')

#     # Input matrix
#     plt.subplot(gs[1])
#     sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_in'), center=0, yticklabels=False, cmap='coolwarm')
#     plt.title('Input layer')

#     # Output matrix
#     plt.subplot(gs[5])
#     sns.heatmap((model_table & {'model_id':  model_id}).fetch1('w_out'), center=0, cmap='coolwarm')
#     plt.title('Output layer')

#     n_trials = trial_table.fetch().shape[0]
#     z = trial_table.fetch('output')
#     correct_choice = trial_table.fetch('correct_choice')
#     context = trial_table.fetch('context')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')
#     x = trial_table.fetch('hidden')

#     id_df = pd.DataFrame([])
#     T = 15
#     for i in range(900):
#         for t in range(15):
#             if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
#                 data = {'time': t*5 ,
#                         'correct_choice': correct_choice[i],
#                         'motion_coh': motion_coh[i],
#                         'color_coh': color_coh[i],
#                         'z': float(z[i][t*5,1] - z[i][t*5,0]),
#                         'context_x': float(x[i][t*5,1] - x[i][t*5,0]),
#                         'motion_x': float(x[i][t*5,3] - x[i][t*5,2]),
#                         'color_x': float(x[i][t*5,5] - x[i][t*5,4]),
#                        'trial':i,
#                        'context': context[i]}

#                 data = pd.DataFrame(data.items())
#                 data = data.transpose()
#                 data.columns = data.iloc[0]
#                 data = data.drop(data.index[[0]])
#                 id_df = id_df.append(data)

#     id_df["correct_choice"] = id_df["correct_choice"].astype(float)
#     id_df["z"] = id_df["z"].astype(float)
#     id_df["time"] = id_df["time"].astype(float)
#     id_df["context_x"] = id_df["context_x"].astype(float)
#     id_df["motion_x"] = id_df["motion_x"].astype(float)
#     id_df["color_x"] = id_df["color_x"].astype(float)
#     id_df["motion_coh"] = id_df["motion_coh"].astype(float)
#     id_df["color_coh"] = id_df["color_coh"].astype(float)

#     plt.subplot(gs[2])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="z",
#     hue="correct_choice",
#     style='context',
#     units="trial",
#     palette=palette,
#     legend='brief',
#     lw=2,
#     estimator=None
#     )
#     plt.legend(loc='upper left')



#     # Panel C (psychometrics)

#     z = trial_table.fetch('output')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')


#     id_df = pd.DataFrame([])
#     for i in range(900):
#             data = {'motion_coh': motion_coh[i],
#                     'color_coh': color_coh[i],
#                     'choice': float(np.maximum(np.sign(z[i][-1,1] - z[i][-1,0]),0)),
#                     'context': context[i]}

#             data = pd.DataFrame(data.items())
#             data = data.transpose()
#             data.columns = data.iloc[0]
#             data = data.drop(data.index[[0]])
#             id_df = id_df.append(data)
#     motion_df = id_df[id_df['context']=="motion"]
#     color_df = id_df[id_df['context']=="color"]

#     motion_df = motion_df.groupby(['motion_coh', 'color_coh'])
#     color_df = color_df.groupby(['motion_coh', 'color_coh'])
#     motion_df = ((motion_df['choice'].apply(np.mean))).reset_index()
#     color_df = ((color_df['choice'].apply(np.mean))).reset_index()


#     plt.subplot(gs[3])
#     sns.heatmap(motion_df.pivot(index='motion_coh', columns='color_coh', values='choice'), annot=True,vmin=0, vmax=1)
#     plt.title('Motion context')

#     plt.subplot(gs[4])
#     sns.heatmap(color_df.pivot(index='motion_coh', columns='color_coh', values='choice'), annot=True,vmin=0, vmax=1)
#     plt.title('Color context')




# # FIGURE FOUR
# def plot_figure_four(lca_table,trial_table):
#     gs = gridspec.GridSpec(2, 4,hspace=.5) 
#     plt.figure(figsize=(24, 8)) 
    
#     n_trials = trial_table.fetch().shape[0]
#     z = trial_table.fetch('output')
#     correct_choice = trial_table.fetch('correct_choice')
#     context = trial_table.fetch('context')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')
#     q = lca_table.fetch1('q')
#     x = np.stack(trial_table.fetch('hidden')) @ q.T

#     id_df = pd.DataFrame([])
#     T = 15
#     for i in range(900):
#         for t in range(15):
#             if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
#                 data = {'time': t*5 ,
#                         'correct_choice': correct_choice[i],
#                         'motion_coh': motion_coh[i],
#                         'color_coh': color_coh[i],
#                         'z': float(z[i][t*5,1] - z[i][t*5,0]),
#                         'context_x_motion': float(x[i][t*5,1] ),
#                         'context_x_color': float(x[i][t*5,0]),
#                         'motion_x_right': float(x[i][t*5,3] ),
#                         'motion_x_left': float( x[i][t*5,2]),
#                         'color_x_right': float(x[i][t*5,5]),
#                         'color_x_left': float (x[i][t*5,4]),
#                         'choice_x_right': float(x[i][t*5,7]),
#                         'choice_x_left': float (x[i][t*5,6]),
#                        'trial':i,
#                        'context': context[i]}

#                 data = pd.DataFrame(data.items())
#                 data = data.transpose()
#                 data.columns = data.iloc[0]
#                 data = data.drop(data.index[[0]])
#                 id_df = id_df.append(data)

#     id_df["correct_choice"] = id_df["correct_choice"].astype(float)
#     id_df["z"] = id_df["z"].astype(float)
#     id_df["time"] = id_df["time"].astype(float)
#     id_df["context_x_motion"] = id_df["context_x_motion"].astype(float)
#     id_df["context_x_color"] = id_df["context_x_color"].astype(float)
#     id_df["motion_x_right"] = id_df["motion_x_right"].astype(float)
#     id_df["motion_x_left"] = id_df["motion_x_left"].astype(float)
#     id_df["color_x_right"] = id_df["color_x_right"].astype(float)
#     id_df["choice_x_right"] = id_df["choice_x_right"].astype(float)
#     id_df["choice_x_left"] = id_df["choice_x_left"].astype(float)
#     id_df["color_x_left"] = id_df["color_x_left"].astype(float)
#     id_df["motion_coh"] = id_df["motion_coh"].astype(float)
#     id_df["color_coh"] = id_df["color_coh"].astype(float)

#     # Motion context
#     plt.subplot(gs[0])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="context_x_motion",
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     #Color context
#     plt.subplot(gs[4])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="context_x_color",
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Motion right
#     plt.subplot(gs[1])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="motion_x_right",
#     hue = 'motion_coh',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Motion left
#     plt.subplot(gs[5])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="motion_x_left",
#     hue = 'motion_coh',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Color right
#     plt.subplot(gs[2])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="color_x_right",
#     hue = 'color_coh',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Color left
#     plt.subplot(gs[6])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="color_x_left",
#     style='context',
#     hue = 'color_coh',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Choice right
#     plt.subplot(gs[3])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="choice_x_right",
#     hue='correct_choice',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Choice left
#     plt.subplot(gs[7])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="choice_x_left",
#     hue='correct_choice',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    

    
    
    
# # FIGURE SIx
# def plot_figure_six(lca_table, model_table):
#     """
#     Plot correlation between Q and W_ext
    
#     """

#     q = lca_table.fetch1('q')
#     w_in = model_table.fetch1('w_in')
#     w_out = model_table.fetch1('w_out')
#     w_ext = np.concatenate((w_in,w_out.T),axis=1)
#     print(q.shape)
#     sns.heatmap(w_ext @ q )
    
    
    
# # Figure Eight
# # FIGURE FOUR
# def plot_figure_eight(q, trial_table):
#     gs = gridspec.GridSpec(1, 4,hspace=.5) 
#     plt.figure(figsize=(24,4)) 
    
#     n_trials = trial_table.fetch().shape[0]
#     z = trial_table.fetch('output')
#     correct_choice = trial_table.fetch('correct_choice')
#     context = trial_table.fetch('context')
#     motion_coh = trial_table.fetch('motion_coh')
#     color_coh = trial_table.fetch('color_coh')
#     x = np.stack(trial_table.fetch('hidden')) @ q

#     id_df = pd.DataFrame([])
#     T = 15
#     for i in range(900):
#         for t in range(15):
#             if (context[i]=="motion" and motion_coh[i]<color_coh[i]) or (context[i]=="color" and motion_coh[i]>color_coh[i]):
#                 data = {'time': t*5 ,
#                         'correct_choice': correct_choice[i],
#                         'motion_coh': motion_coh[i],
#                         'color_coh': color_coh[i],
#                         'context_x': float(x[i][t*5,0] ),
#                         'motion_x': float(x[i][t*5,1] ),
#                         'color_x': float(x[i][t*5,2]),
#                         'choice_x': float(x[i][t*5,3]),
#                        'trial':i,
#                        'context': context[i]}

#                 data = pd.DataFrame(data.items())
#                 data = data.transpose()
#                 data.columns = data.iloc[0]
#                 data = data.drop(data.index[[0]])
#                 id_df = id_df.append(data)

#     id_df["correct_choice"] = id_df["correct_choice"].astype(float)

#     id_df["time"] = id_df["time"].astype(float)
#     id_df["context_x"] = id_df["context_x"].astype(float)

#     id_df["motion_x"] = id_df["motion_x"].astype(float)

#     id_df["color_x"] = id_df["color_x"].astype(float)
#     id_df["choice_x"] = id_df["choice_x"].astype(float)

#     id_df["motion_coh"] = id_df["motion_coh"].astype(float)
#     id_df["color_coh"] = id_df["color_coh"].astype(float)

#     # context
#     plt.subplot(gs[0])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="context_x",
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     #Color 
#     plt.subplot(gs[1])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="color_x",
#     hue='color_coh',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Motion 
#     plt.subplot(gs[2])
#     palette = sns.color_palette("coolwarm", 6)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="motion_x",
#     hue = 'motion_coh',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')
    
#     # Choice
#     plt.subplot(gs[3])
#     palette = sns.color_palette("coolwarm", 2)
#     sns.lineplot(
#     data=id_df,
#     x="time",
#     y="choice_x",
#     hue = 'correct_choice',
#     style='context',
#     palette=palette,
#     legend='brief',
#     )
#     plt.legend(loc='upper left')



def plot_psychometric(model_id, ax1, ax2):
    """
    Plot psychometric heatmaps for each context for a given model.
    """

    t = 3000
    dt = .2 * 200
    n_t = int(round(t / dt))
    trial_events = {'n_t': int(round(n_t)),
                    'cue_on': int(round(n_t * .1)),
                    'cue_off': int(round(n_t * .33)),
                    'stim_on': int(round(n_t * .4)),
                    'stim_off': int(round(n_t)),
                    'dec_on': int(round(n_t * .75)),
                    'dec_off': int(round(n_t))}

    # Load inputs and labels
    inputs, labels, mask, conditions = generate_trials(**trial_events,
                                              n_trials=15,
                                              alpha=0.2,
                                              tau=200,
                                              sigma_in=.01,
                                              baseline=0.2,
                                              n_coh=6)


    n = (Model() & {'model_id':model_id}).fetch1('n')
    size = (Model() & {'model_id':model_id}).fetch1('connectivity')
    rnn = RNNModule(connectivity=size, mask=mask, n=n)

    rnn.recurrent_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_rec'))
    rnn.input_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_in'))
    rnn.output_layer.weight.data = torch.tensor((Model() & {'model_id':model_id}).fetch1('w_out'))

    z_mask, z, x = rnn.forward(inputs)
    z = z.detach().cpu().numpy()


    data = np.concatenate((pd.DataFrame(conditions).values,z[:,-1,1,None]-z[:,-1,0,None]),1)
    df = pd.DataFrame(data=data, columns=['context','motion_coh','color_coh','correct_choice','output'])
  
  
    motion_df = df[df['context']=='motion'].groupby(['motion_coh', 'color_coh'])
    color_df = df[df['context']=='color'].groupby(['motion_coh', 'color_coh'])
    
    motion_df = ((motion_df['output'].apply(np.mean))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')
    color_df = ((color_df['output'].apply(np.mean))).reset_index().pivot(index='motion_coh', columns='color_coh', values='output')

    
    sns.heatmap(motion_df, annot=True,vmin=-1.2, vmax=1.2,center=0, ax = ax1, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2))
    plt.title('Motion context')

    sns.heatmap(color_df, annot=True,vmin=-1.2, vmax=1.2,center=0, ax = ax2, xticklabels=motion_df.columns.values.round(2),yticklabels=motion_df.index.values.round(2))
    plt.title('Color context')
    
    
    
    
def plot_ensemble_stats(model_id,p):

    fig = plt.figure(figsize=(40,6))

    gs = gridspec.GridSpec(1,4, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    #ax4 = fig.add_subplot(gs[4])
    #ax3 = fig.add_subplot(gs[3])

    df=pd.DataFrame((LCA() & {'model_id': model_id}).fetch())
    df['valid_loss']=df['valid_loss'].astype(float)
    df['train_loss']=df['train_loss'].astype(float)
    # sns.scatterplot(data=df,x='epochs',y='valid_loss',color='gray',ax=ax0)
    # ax0.set_title('Overtraining')

    # sns.swarmplot(data=df,x='patience',y='epochs', color='gray',ax=ax1)
    # ax1.set_title('Patience and epochs')

    sns.swarmplot(data=df,x='patience',y='valid_loss', color='grey',ax=ax0)
    ax0.set_ylabel('Test error')

    # df1 = df.groupby('patience').apply(lambda x: total_variance(x)).reset_index()
    # sns.barplot(data=df1,x='patience',y=0, color='gray',ax=ax1,palette='gist_gray_r')
    # ax1.set_title('Patience and sloppiness')
    # ax1.set_ylabel('Sloppiness')

    df['overfitting'] = df['valid_loss'] - df['train_loss']
    sns.swarmplot(data=df,x='patience',y='overfitting',ax=ax1,color='gray')
    ax1.set_ylabel('Test - train error')

    data=[]
    for patience in np.unique(df['patience']):
        df = pd.DataFrame((LCA() & {'model_id': model_id} & 'weight_decay=0' & {'patience':patience} ).fetch())
        percentile = np.percentile(df['valid_loss'].astype(float),p)
        w_recs = np.stack(df[df['valid_loss']<percentile]['w_rec']).reshape(-1,64)
        pca = PCA()
        pca.fit(w_recs)
        #plt.plot((pca.explained_variance_[:10]),marker='.',lw=4,markersize=25,label='0')
        for component in range(10):
            data.append({'Principal component':component,'patience':patience,'variance':np.cumsum(pca.explained_variance_)[component],'ratio':np.cumsum(pca.explained_variance_ratio_)[component]})
    df = pd.DataFrame(data)   
    sns.lineplot(data=df,x='Principal component',y='variance',hue='patience',lw=2,ax=ax2,legend=False,palette='gist_gray_r')
    ax2.set_ylabel('Explained variance ')
    sns.scatterplot(data=df,x='Principal component',y='variance',hue='patience',s=100,ax=ax2,legend=False,palette='gist_gray_r')

    sns.lineplot(data=df,x='Principal component',y='ratio',hue='patience',lw=2,ax=ax3,legend=False,palette='gist_gray_r')
    ax3.set_ylabel('Explained variance ratio')
    ax3.set_ylim(0,1)
    sns.scatterplot(data=df,x='Principal component',y='ratio',hue='patience',s=100,ax=ax3,legend=False,palette='gist_gray_r')
    
    
    
    
    
# def plot_handmade_circuit():
#     fig = plt.figure(figsize=(8,4))

#     gs0 = gridspec.GridSpec(2, 1, figure=fig,height_ratios=[5,4])

#     # Bottom row
#     gs01 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[1])
#     ax1 = fig.add_subplot(gs01[0])
#     ax2 = fig.add_subplot(gs01[1])
#     ax3 = fig.add_subplot(gs01[2])
#     ax4 = fig.add_subplot(gs01[3])

#     # Top row
#     gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[1,1.25,1], wspace=.25)

#     gs000 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs00[0], width_ratios=[5,1],height_ratios=[5,1])
#     gs001 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs00[1], width_ratios=[8,6],height_ratios=[8,2],wspace=0.1)
#     gs002 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs00[2], width_ratios=[1,3,1],height_ratios=[1,3,1])

#     # network graph
#     ax5 = fig.add_subplot(gs00[0])

#     # solution
#     ax8 = fig.add_subplot(gs001[0, 0])

#     ax9 = fig.add_subplot(gs001[0, 1])
#     ax10 = fig.add_subplot(gs001[1, 0])

#     # decomposition of mechanism
#     ax11 = fig.add_subplot(gs002[0, 0])
#     ax12 = fig.add_subplot(gs002[0, 1])
#     ax13 = fig.add_subplot(gs002[0, 2])
#     ax14 = fig.add_subplot(gs002[1, 0])
#     ax15 = fig.add_subplot(gs002[1, 1])
#     ax16 = fig.add_subplot(gs002[1, 2])
#     ax17 = fig.add_subplot(gs002[2, 0])
#     ax18 = fig.add_subplot(gs002[2, 1])
#     ax19 = fig.add_subplot(gs002[2, 2])

#     q = np.eye(8)
#     q =  q[[0,2,4,6],:] - q[[1,3,5,7],:]
#     plot_sorted_projections("LZCI7OM8",q,ax1, ax2, ax3, ax4)

#     plot_model_parameters(Model(),"LZCI7OM8", ax8, ax9, ax10)

#     plot_circuit_decomposition(Model(), "LZCI7OM8", ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19)

#     plot_circuit_network(Model(), "LZCI7OM8", ax = ax5)
#     ax5.spines["top"].set_visible(False)

#     plt.savefig('/users/langdon/figures/circuitpaperfigures/Figure_1.png',dpi=1000, transparent=True)