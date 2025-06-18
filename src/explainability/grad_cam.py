import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        try:
            # Reset
            self.activations = None
            self.gradients = None
            
            # Ensure input requires gradients
            input_tensor = input_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target_score = output[0, class_idx]
            target_score.backward()
            
            # Check if hooks captured data
            if self.activations is None or self.gradients is None:
                raise ValueError("Failed to capture activations or gradients")
            
            # Process on CPU
            gradients = self.gradients.cpu().data.numpy()
            activations = self.activations.cpu().data.numpy()
            
            # Get first sample from batch
            gradients = gradients[0]  # [C, H, W]
            activations = activations[0]  # [C, H, W]
            
            # Global average pooling on gradients
            weights = np.mean(gradients, axis=(1, 2))  # [C]
            
            # Weighted combination of activation maps
            cam = np.sum(weights.reshape(-1, 1, 1) * activations, axis=0)
            
            # Apply ReLU
            cam = np.maximum(cam, 0)
            
            # Resize to input size
            cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
            
            # Normalize
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            return cam
            
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()


class SimpleGradCAM:
    """Simplified GradCAM that works with common architectures"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # We'll modify the model temporarily to capture intermediate outputs
        activations = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hook on target layer
        handle = self.target_layer.register_forward_hook(forward_hook('target'))
        
        try:
            # Forward pass
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            # Get the target activation
            target_activation = activations['target']
            
            # Compute gradients
            self.model.zero_grad()
            target_score = output[0, class_idx]
            
            # Use autograd.grad with allow_unused=True
            grads = torch.autograd.grad(
                outputs=target_score,
                inputs=target_activation,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            
            if grads[0] is None:
                # If direct gradient computation fails, try alternative approach
                return self._alternative_gradcam(input_tensor, output, class_idx, target_activation)
            
            gradients = grads[0]
            
            # Convert to numpy
            gradients_np = gradients.detach().cpu().numpy()[0]  # [C, H, W]
            activations_np = target_activation.detach().cpu().numpy()[0]  # [C, H, W]
            
            # Compute CAM
            weights = np.mean(gradients_np, axis=(1, 2))
            cam = np.sum(weights.reshape(-1, 1, 1) * activations_np, axis=0)
            
            # Post-process
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
            
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            return cam
            
        finally:
            handle.remove()

    def _alternative_gradcam(self, input_tensor, output, class_idx, target_activation):
        """Alternative method using backward hooks"""
        gradients_list = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients_list.append(grad_output[0])
        
        # Register backward hook
        handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Backward pass
            self.model.zero_grad()
            target_score = output[0, class_idx]
            target_score.backward()
            
            if not gradients_list:
                raise ValueError("Could not capture gradients")
            
            gradients = gradients_list[0]
            
            # Convert to numpy
            gradients_np = gradients.detach().cpu().numpy()[0]
            activations_np = target_activation.detach().cpu().numpy()[0]
            
            # Compute CAM
            weights = np.mean(gradients_np, axis=(1, 2))
            cam = np.sum(weights.reshape(-1, 1, 1) * activations_np, axis=0)
            
            # Post-process
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
            
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            return cam
            
        finally:
            handle.remove()


# Most robust implementation
class RobustGradCAM:
    """Most robust GradCAM implementation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Create a new input tensor that requires gradients
        input_var = input_tensor.clone().detach().requires_grad_(True)
        
        # Store activations and gradients
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        # Register hooks
        fhook = self.target_layer.register_forward_hook(forward_hook)
        bhook = self.target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Forward pass
            output = self.model(input_var)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            score = output[0][class_idx]
            score.backward()
            
            # Check if we got the data
            if activations is None or gradients is None:
                raise ValueError(f"Failed to capture data. Activations: {activations is not None}, Gradients: {gradients is not None}")
            
            # Convert to numpy
            act_np = activations.detach().cpu().numpy()[0]  # Remove batch dimension
            grad_np = gradients.detach().cpu().numpy()[0]   # Remove batch dimension
            
            # Compute weights (global average pooling of gradients)
            weights = np.mean(grad_np, axis=(1, 2))
            
            # Compute weighted combination
            cam = np.zeros(act_np.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * act_np[i, :, :]
            
            # Apply ReLU
            cam = np.maximum(cam, 0)
            
            # Resize to input size
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            cam = cv2.resize(cam, (w, h))
            
            # Normalize
            if cam.max() != cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            return cam
            
        finally:
            fhook.remove()
            bhook.remove()