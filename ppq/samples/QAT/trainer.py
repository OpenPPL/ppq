from typing import Iterable, Tuple

import torch
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph, TrainableGraph
from tqdm import tqdm


class ImageNetTrainer():
    """
    ### Network Trainer for ImageNet Classification Task.

    Member Functions:
    1. epoch(): do one epoch training.
    2. step(): do one step training.
    3. eval(): evaluation on given dataset.
    4. save(): save trained model.
    5. clear(): clear training cache.

    PPQ will create a TrainableGraph on you graph, a wrapper class that
        implements a set of useful functions that enable training. You are recommended
        to edit its code to add new feature on graph.

    Optimizer controls the learning process and determine the parameters values ends up learning,
        you can rewrite the defination of optimizer and learning scheduler in __init__ function.
        Tuning them carefully, as hyperparameters will greatly affects training result.
    """
    
    def __init__(
        self, graph: BaseGraph, device: str = 'cuda') -> None:

        self._epoch        = 0
        self._step         = 0
        self._best_metric  = 0
        self._loss_fn      = torch.nn.CrossEntropyLoss().to(device)
        self._executor     = TorchExecutor(graph, device=device)
        self._training_graph = TrainableGraph(graph)
        self.graph           = graph

        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True

        self._optimizer = torch.optim.RAdam(
            params=self._training_graph.parameters(), lr=3e-5)
        self._lr_scheduler = None

    def epoch(self, dataloader: Iterable) -> float:
        """Do one epoch Training with given dataloader.
        
        Given dataloader is supposed to be a iterable container of batched data,
            for example it can be a list of [img(torch.Tensor), label(torch.Tensor)].
        
        If your data has other layout that is not supported by this function,
            then you are supposed to rewrite the logic of epoch function by yourself.
        """
        epoch_loss = 0
        for bidx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {self._epoch}: ', total=len(dataloader))):
            if type(batch) not in {tuple, list}:
                raise TypeError('Feeding Data is invalid, expect a Tuple or List like [data, label], '
                                f'however {type(batch)} was given. To feed customized data, you have to rewrite '
                                '"epoch" function of PPQ Trainer.')
            if len(batch) != 2:
                raise ValueError('Unrecognized data format, '
                                 'your dataloader should contains batched data like [data, label]')
            data, label = batch
            data, label = data.cuda(), label.cuda()

            _, loss = self.step(data, label, True)
            epoch_loss += loss

        self._epoch += 1
        print(f'Epoch Loss: {epoch_loss / len(dataloader):.4f}')
        return epoch_loss

    def step(self, data: torch.Tensor, label: torch.Tensor, training: bool) -> Tuple[torch.Tensor, float]:
        """Do one step Training with given data(torch.Tensor) and label(torch.Tensor).
        
        This one-step-forward function assume that your model have only one input and output variable.
        
        If the training model has more input or output variable, then you might need to
            rewrite this function by yourself.
        """
        if training:
            pred = self._executor.forward_with_gradient(data)[0]
            loss = self._loss_fn(pred, label)
            
            loss.backward()
            if self._lr_scheduler is not None: 
                self._lr_scheduler.step(epoch=self._epoch)
            self._optimizer.step()
            self._training_graph.zero_grad()
        
            self._step += 1
            return pred, loss.item()

        else:
            pred = self._executor.forward(data)[0]
            loss = self._loss_fn(pred, label)

            return pred, loss.item()

    def eval(self, dataloader: Iterable) -> float:
        """Do Evaluation process on given dataloader.
        
        Split your dataset into training and evaluation dataset at first, then
            use eval function to monitor model performance on evaluation dataset.

        Here are some options to prevent overfitting, which helps improve the model performance.
        1. Train with more data.
        2. Data augmentation.
        3. Addition of noise to the input data.
        """
        total_pred, total_correct = 0, 0
        with torch.no_grad():
            for bidx, batch in enumerate(tqdm(dataloader, desc='Eval: ')):
                if type(batch) not in {tuple, list}:
                    raise TypeError('Feeding Data is invalid, expect a Tuple or List like [data, label], '
                                    f'however {type(batch)} was given. To feed customized data, you have to rewrite '
                                    '"eval" function of PPQ Trainer.')
                if len(batch) != 2:
                    raise ValueError('Unrecognized data format, '
                                     'your dataloader should contains batched data like [data, label]')
                data, label = batch
                data, label = data.cuda(), label.cuda()

                pred, _ = self.step(data, label, False)
                pred_label = torch.argmax(pred, dim=-1)
                total_correct += torch.sum(pred_label == label).item()
                total_pred    += pred_label.numel()

        print(f'Classification Accuracy: {total_correct / total_pred * 100:.3f}%\n')
        return total_correct / total_pred

    def save(self, file_path: str):
        """ Save model to given path.
        Saved model can be read by ppq.api.load_native_model function.
        """
        from ppq.parser import NativeExporter
        exporter = NativeExporter()
        exporter.export(file_path=file_path, graph=self.graph)

    def clear(self):
        """Clear training state."""
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = False
            tensor._grad = None
