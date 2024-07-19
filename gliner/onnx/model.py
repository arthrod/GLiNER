from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
import onnxruntime as ort
import numpy as np
import torch

from ..config import GLiNERConfig
from ..modeling.base import GLiNERModelOutput

class BaseORTModel(ABC):
    def __init__(self, config: GLiNERConfig, session: ort.InferenceSession):
        self.config = config
        self.session = session
        self.providers = session.get_providers()
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.required_outputs = ['logits']

        if 'CUDAExecutionProvider' in self.providers[0]:
            self.use_io_binding = True
        else:
            self.use_io_binding = False

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for ONNX model inference.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input names and tensors.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of input names and numpy arrays.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary of input names and tensors.")
        
        prepared_inputs = {}
        for key, tensor in inputs.items():
            if key not in self.input_names:
                warnings.warn(f"Input key '{key}' not found in ONNX model's input names. Ignored.")
                continue
            prepared_inputs[key] = tensor.cpu().detach().numpy()
        return prepared_inputs

    @abstractmethod
    def prepare_iobinding_output(self, inputs) -> Dict[str, np.ndarray]:
        """ Prepare empty outputs in the case of iobinding inference"""
        pass 
    
    def run_with_iobinding(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run the ONNX model inference with iobinding in the case of CUDAExecutionProvider.
        Args:
        inputs (Dict[str, np.ndarray]): Prepared inputs for the model.
        Returns:
        Dict[str, np.ndarray]: Model's outputs as numpy arrays.
        """
        io_binding = self.session.io_binding()
        
        # Bind inputs
        for name, array in inputs.items():
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(array, 'cuda', 0)
            io_binding.bind_ortvalue_input(name, input_ortvalue)
        
        # Prepare and bind outputs
        outputs = self.prepare_iobinding_output(inputs)
        for name in self.required_outputs:
            if name in outputs:
                output_ortvalue = ort.OrtValue.ortvalue_from_numpy(outputs[name], 'cuda', 0)
                io_binding.bind_ortvalue_output(name, output_ortvalue)
        
        # Run options
        ro = ort.RunOptions()
        ro.add_run_config_entry("gpu_graph_id", "0")  # Optional if using only one CUDA graph
        
        # Run inference
        self.session.run_with_iobinding(io_binding, ro)
        
        # Convert outputs back to numpy arrays
        for name in self.required_outputs:
            outputs[name] = io_binding.get_outputs()[self.output_names[name]].numpy()
        
        return outputs
    
    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run the ONNX model inference.
        
        Args:
            inputs (Dict[str, np.ndarray]): Prepared inputs for the model.
        
        Returns:
            Dict[str, np.ndarray]: Model's outputs as numpy arrays.
        """
        if self.use_io_binding:
            outputs = self.run_with_iobinding(inputs)
        else:
            onnx_outputs = self.session.run(None, inputs)
            outputs = {name: onnx_outputs[idx] for name, idx in self.output_names.items()}
        return outputs

    @abstractmethod
    def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to perform forward pass. Must be implemented by subclasses.
        """
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class SpanORTModel(BaseORTModel):
    def prepare_iobinding_output(self, inputs) -> Dict[str, np.ndarray]:
        """ Prepare empty outputs in the case of iobinding inference"""
        input_ids = inputs['input_ids']            
        batch_size = input_ids.shape[0]
        text_length = inputs['text_lengths'].max()
        num_spans = self.config.max_width
        max_classes = (input_ids == self.config.class_token_index).sum(axis=1).max()

        logits = np.empty((batch_size, text_length, num_spans, max_classes))
        return {'logits': logits}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                span_idx: torch.Tensor, span_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for span model using ONNX inference.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            span_idx (torch.Tensor): Span indices tensor.
            span_mask (torch.Tensor): Span mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
            'span_idx': span_idx,
            'span_mask': span_mask
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERModelOutput(
            logits=inference_output['logits']
        )
        return outputs

class TokenORTModel(BaseORTModel):
    def prepare_iobinding_output(self, inputs) -> Dict[str, np.ndarray]:
        """ Prepare empty outputs in the case of iobinding inference"""
        input_ids = inputs['input_ids']            
        batch_size = input_ids.shape[0]
        text_length = inputs['text_lengths'].max()
        max_classes = (input_ids == self.config.class_token_index).sum(axis=1).max()

        logits = np.empty((batch_size, text_length, max_classes))
        return {'logits': logits}
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass for token model using ONNX inference.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERModelOutput(
            logits=inference_output['logits']
        )
        return outputs