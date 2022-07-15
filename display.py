import visdom
from typing import Optional, List, Any, Union, Mapping, overload, Text
from visdom import Visdom

import numpy as np

import torch

from os.path import join

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

import subprocess

def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    if split_by_img:
        n = pred_vessels.shape[0]
        return (np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]),
                np.array([pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]))
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    result=result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_gpu_temperature_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=temperature.gpu',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_temperature = [int(x) for x in result.strip().split('\n')]
    gpu_temperature_map = dict(zip(range(len(gpu_temperature)), gpu_temperature))
    return gpu_temperature_map

def get_gpu_name():
    result = subprocess.check_output(
        [
            'nvidia-smi', '-L'
        ]
    )
    return result.decode('utf-8')

def get_nvidia_smi():
    result=subprocess.check_output([
        'nvidia-smi'
    ])
    return result.decode('utf-8')

def get_disk_usage():
    result=subprocess.check_output([
        'df','-h','/etc'
    ])
    return result.decode('utf-8')


# TODO: make all functions avaliable in visdomE, especially on net part.
### Type aliases for commonly-used types.
# For optional 'options' parameters.
# The options parameters can be strongly-typed with the proposed TypedDict type once that is incorporated into the standard.
# See  http://mypy.readthedocs.io/en/latest/more_types.html#typeddict.
_OptOps = Optional[Mapping[Text, Any]]
_OptStr = Optional[Text]  # For optional string parameters, like 'window' and 'env'.

# No widely-deployed stubs exist at the moment for torch or numpy. When they are available, the correct type of the tensor-like inputs
# to the plotting commands should be
# Tensor = Union[torch.Tensor, numpy.ndarray, List]
# For now, we fall back to 'Any'.
Tensor = Any

# The return type of 'Visdom._send', which is turn is also the return type of most of the the plotting commands.
# It technically can return a union of several different types, but in normal usage,
# it will return a single string. We only type it as such to prevent the need for users to unwrap the union.
# See https://github.com/python/mypy/issues/1693.
_SendReturn = Text
'''
The main difference between Visdom and Visdom_E is that Vidsom_E can leave out the annoying counter and will automatically
judge whether it's the first time to paint a line or Image. As for Image, it support more type like PIL.Image and it
could automatically transform the channel order like CHW to HWC.
In the future I will develope more characters like performance analyze heatmap and tsne. 
'''
#use_incoming_socket=False
class Visdom_E(Visdom):
    def __init__(
            self,env=None
    ):
        if env!=None:
            super(Visdom_E, self).__init__(env=env)
        else:
            super(Visdom_E, self).__init__()
        self.clockDict={}

    def set(self,name,env=None):
        assert self.clockDict.get(name) == None
        self.clockDict[name] = {'clock': np.array([0]), 'env': env, 'win': None,'opt':name}

    def resetClock(self,name):
        assert name in self.clockDict
        self.clockDict[name]['clock']=np.array([0])

    def lineE(
        self,
        Y: Tensor,
        name:str,
        noAdd=False,
        withName=None,
        legend=None
    ):
        if isinstance(withName,type(None)):
            withName=name
        if isinstance(legend,type(None)):
            d=dict(title=self.clockDict[name]['opt'])
        else:
            d=dict(title=self.clockDict[name]['opt'],legend=list(legend))
        if isinstance(self.clockDict[name]['win'],type(None)):
            self.clockDict[name]['win']=super().line(Y,X=self.clockDict[name]['clock'],
                                                         env=self.clockDict[name]['env'],name=withName,opts=d)
        else:
            super().line(Y,X=self.clockDict[name]['clock'],win=self.clockDict[name]['win'],
                                                         env=self.clockDict[name]['env'],update='append',name=withName,opts=d)

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def imageE(
            self,
            img: Tensor,
            name: str,
            noAdd=False,
            needRegular=False
    ):
        if needRegular:
            img=(img-torch.min(img))/(torch.max(img)-torch.min(img)+1e-12)
        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = super().image(img,env=self.clockDict[name]['env'],opts=dict(title=self.clockDict[name]['opt']))

        else:
            super().image(img,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'],opts=dict(title=self.clockDict[name]['opt']))

        if not noAdd:
            self.clockDict[name]['clock'] += 1
             

    def tsne(self,
             batchXchannels,
             batchlabels,
             win=None,
             env=None,
             name=None,
             colors=None
             ):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        cls=np.unique(batchlabels)
        print('Unique classes',str(cls))
        new_bc=[]
        for i in range(len(cls)):
            if (batchlabels==i).nonzero().shape[0]<100:
                new_bc.append(batchXchannels[batchlabels==i])
            else:
                new_bc.append(batchXchannels[batchlabels==i][:100])
        batchXchannels=np.concatenate(new_bc)
        # plot_only = 1000
        
        low_dim_embs = tsne.fit_transform(batchXchannels[:, :])
        labels = batchlabels[:]
        plt.cla()
        if colors is None:
            colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'purple']
        else:
            colors=colors/255
        X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
        for x, y, s in zip(X, Y, labels):
            # if s==0:
            #     continue
            c = cm.rainbow(int(255 * s // 9))
            # plt.text(x, y, s, backgroundcolor=c, fontsize=9)

            if type(colors)==np.ndarray:
                plt.scatter(x, y, c=colors[s].reshape(1,-1), s=16, lw=0)
            else:
                plt.scatter(x, y, c=colors[s], s=16, lw=0)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        if name==None:
            plt.title('t-SNE')
        else:
            plt.title(name)
        return self.matplot(plt,win=win,env=env)

    def tsneE(self,
        batchXchannels,
        batchlabels,
        name:str,
        noAdd=False,
        colors=None,
    ):
        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = self.tsne(batchXchannels,batchlabels,env=self.clockDict[name]['env'],name=name,colors=colors)
        else:
            self.tsne(batchXchannels,batchlabels,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'],name=name,colors=colors)

        if not noAdd:
            self.clockDict[name]['clock'] += 1
            
    def matplotE(self,
        mat,
        name:str,
        noAdd=False
    ):
        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = self.matplot(mat,env=self.clockDict[name]['env'])
        else:
            self.matplot(mat,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'])
        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def gpuDisk(
            self,
            name: str,
            noAdd=False
    ):
        try:
            gpu = get_nvidia_smi()
            disk = get_disk_usage()
            display = (gpu + '\n' + disk)
            display = display.replace('\n', '<br>')
        except:
            display="Warning: You have called gpuDisk in other place at the same time."

        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = super().text(display,env=self.clockDict[name]['env'])

        else:
            super().text(display,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'])

        if not noAdd:
            self.clockDict[name]['clock'] += 1

    def textE(
            self,
            data:dict,
            name: str,
            noAdd=False
    ):
        display=""
        for key in data:
            display+="%s "%key
            display+=str(data[key])
            display+='<br>'

        if self.clockDict[name]['clock'] == 0:
            self.clockDict[name]['win'] = super().text(display,env=self.clockDict[name]['env'])

        else:
            super().text(display,win=self.clockDict[name]['win'],env=self.clockDict[name]['env'])

        if not noAdd:
            self.clockDict[name]['clock'] += 1
    
    #add measure functions
    def PRCurve(self,
                precision:np.ndarray,
                recall:np.ndarray,
                name:str,
                visName:str,
                plotStyle: str,
                bestPoint=None,
                savePath=None,
                noAdd=False):
        plt.cla()
        plt.title("Precision Recall Curve")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        if not isinstance(bestPoint,type(None)):
            plt.plot(recall, precision, label=name)
            plt.legend(loc='lower left')
        plt.plot(bestPoint[1], bestPoint[0], plotStyle, label=name + ' best f1')
        plt.legend(loc='lower left')

        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = self.matplot(plt,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        else:
            self.matplot(plt,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1
        if not isinstance(savePath,type(None)):
            plt.savefig(join(savePath, 'PRCurve.png'))


    def ROCCurve(self,
                 predict:np.ndarray,
                 groundTruth:np.ndarray,
                 name:str,
                 visName:str,
                 mask=None,
                 savePath=None,
                 noAdd=False):
        if not isinstance(mask,type(None)):
            groundTruth,predict=pixel_values_in_mask(groundTruth,predict,mask)
        plt.cla()
        plt.title("ROC Curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        fpr, tpr, _ = roc_curve(groundTruth.flatten(), predict.flatten())
        plt.plot(fpr, tpr, label=name)
        plt.legend(loc='lower left')
        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = self.matplot(plt, win=self.clockDict[visName]['win'],
                                                          env=self.clockDict[visName]['env'])

        else:
            self.matplot(plt, win=self.clockDict[visName]['win'], env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1
        if not isinstance(savePath,type(None)):
            plt.savefig(join(savePath, 'ROCCurve.png'))


    def progressBar(
            self,
            percent: int,
            name: str,
            visName: str,
            noAdd=False
    ):
        barHTML='''<!DOCTYPE html>
                    <html>
                    <body>
                    
                    <h1>{}</h1>
                    
                    <div style="width: 100%;background-color: #ddd;text-align:center;">
                      <div style="width: {}%;height: 30px;background-color: #4CAF50;text-align: center;line-height: 30px;color: white;">{}%</div>
                    </div>
                    
                    </body>
                    </html>'''.format(name,percent,percent)
        if self.clockDict[visName]['clock'] == 0:
            self.clockDict[visName]['win'] = super().text(barHTML,env=self.clockDict[visName]['env'])

        else:
            super().text(barHTML,win=self.clockDict[visName]['win'],env=self.clockDict[visName]['env'])

        if not noAdd:
            self.clockDict[visName]['clock'] += 1

    def plot_confusion_matrix(self,name,cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          ):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        save_path:    path to save image.

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                                [  604,  4392,  6233],
                                                [  162,  2362, 31760]]),
                        normalize    = False,
                        target_names = ['high', 'medium', 'low'],
                        title        = "Confusion Matrix")

        """
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools
        matplotlib.use('Agg')
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.cla()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        # plt.savefig(save_path)
        self.matplotE(plt,name)
    
    def plot_confusion_matrixs(self,name,cms,
                          cmap=None,
                          normalize=True,
                          ):
        plot_confusion_matrixs_(cms,cmap=cmap,normalize=normalize)
        self.matplotE(plt,name)

def plot_confusion_matrixs_(cms,
                          cmap=None,
                          normalize=True,
                          ):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cms:           multiple confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        save_path:    path to save image.

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                                [  604,  4392,  6233],
                                                [  162,  2362, 31760]]),
                        normalize    = False,
                        target_names = ['high', 'medium', 'low'],
                        title        = "Confusion Matrix")

        """
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools
        matplotlib.use('Agg')

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        t=cms.shape[0]

        plt.figure(figsize=(8*t, 6))
        plt.cla()

        for i in range(t):
            cm=cms[i]
            plt.subplot(1,t,i+1)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            # plt.title(title)
            # plt.colorbar()

            # accuracy = np.trace(cm) / float(np.sum(cm))
            # misclass = 1 - accuracy

            # if target_names is not None:
                # tick_marks = np.arange(len(target_names))
                # plt.xticks(tick_marks, target_names, rotation=45)
                # plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


            # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            #     if normalize:
            #         plt.text(j, i, "{:0.4f}".format(cm[i, j]),
            #                 horizontalalignment="center",
            #                 color="white" if cm[i, j] > thresh else "black")
            #     else:
            #         plt.text(j, i, "{:,}".format(cm[i, j]),
            #                 horizontalalignment="center",
            #                 color="white" if cm[i, j] > thresh else "black")


            
            # plt.ylabel('True label')
            # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        # plt.savefig(save_path)
        plt.tight_layout()
        # plt.show()
        return plt
        # self.matplotE(plt,name)

if __name__=='__main__':
    import time
    # vis=Visdom_E()
    # vis.set('progress',env='main')
    # for i in range(100):
    #     vis.progressBar(i,'test','progress')
    #     time.sleep(1)
    import numpy as np
    a=np.random.randn(9,30,30)
    plot_confusion_matrixs_(a)

