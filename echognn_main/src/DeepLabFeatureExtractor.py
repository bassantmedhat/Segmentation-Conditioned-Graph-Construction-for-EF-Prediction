# changed load video function as i called cv2 convert color and changed the dimesnsions to the 
# input segmentation model will need to change them in the data set and make check of the outout shape
# to the graph model 


from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize

import cv2
import numpy as np 
import os 
import scipy.signal
import skimage.draw
# class DeepLabFeatureExtractor(nn.Module):
#     def __init__(self, model, layer_number):
#         super(DeepLabFeatureExtractor, self).__init__()
#         self.backbone = model.backbone  # Encoder
#         self.classifier = model.classifier  # Decoder
#         self.layer_number = layer_number
#         # print(model) 

#     def forward(self, x):
#         encoder_features = []
#         decoder_features = []
        
#         # Extract features from backbone (encoder)
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)
        
#         for name, layer in self.backbone.named_children():
#             if "layer" in name:
#                 x = layer(x)
#                 if self.layer_number in name:
#                     encoder_features.append(x)
        
#         # Decoder forward pass
#         decoder_output = self.classifier(x)
#         decoder_features.append(decoder_output)
#         # print(f"decoder_features:{decoder_features.shape}, encoder_features:{encoder_features.shape}")
        
        
#         #  Upsampling (if required)
#         decoder_output = torch.nn.functional.interpolate(decoder_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
#         decoder_features.append(decoder_output)
        
#         return decoder_features, encoder_features

#         # return decoder_features, encoder_features

# def print_shapes(model, input_tensor):
#     # A dictionary to store the layer-wise input and output shapes
#     layer_shapes = []

#     def hook_fn(module, input, output):
#         # Record the input and output shapes
#         input_shape = tuple(input[0].shape)  # Assuming single input tensor
#         output_shape = tuple(output.shape)
#         layer_shapes.append((module.__class__.__name__, input_shape, output_shape))
    
#     # Register the hook for each layer
#     hooks = []
#     for layer in model.children():
#         hook = layer.register_forward_hook(hook_fn)
#         hooks.append(hook)
    
#     # Forward pass to capture the shapes
#     model(input_tensor)
    
#     # Remove hooks to avoid memory leaks
#     for hook in hooks:
#         hook.remove()
    
#     # Print the table of layer shapes
#     print(f"{'Layer Name':<25} {'Input Shape':<40} {'Output Shape':<40}")
#     print("="*110)
#     for layer_name, input_shape, output_shape in layer_shapes:
#         print(f"{layer_name:<25} {str(input_shape):<40} {str(output_shape):<40}")
    

class DeepLabFeatureExtractor(nn.Module):
    def __init__(self, model, layer_number):
        super(DeepLabFeatureExtractor, self).__init__()
        self.backbone = model.backbone  # Encoder
        self.classifier = model.classifier  # Decoder
        self.layer_number = layer_number

    def forward(self, x):
        encoder_features = []
        decoder_features = []
        
        # # Extract features from backbone (encoder)
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        
        for name, layer in self.backbone.named_children():
            # print(name)
            # if "layer" in name:
                x = layer(x)
                if self.layer_number in name:
                    encoder_features.append(x)
        
        # Decoder forward pass
        decoder_output = self.classifier(x)
        decoder_features.append(decoder_output)
    
        
        return decoder_features, encoder_features
import torch
import torchvision
import torch.nn as nn

# Load the pretrained DeepLabV3 model

# Modify the model if necessary to capture features at layer 4
class DeepLabV3WithFeatures(nn.Module):
    def __init__(self, model):
        super(DeepLabV3WithFeatures, self).__init__()
        self.backbone = model.backbone
        self.classifier = model.classifier

    def forward(self, x):
        input_size = x.shape[-2:]  
        features = self.backbone(x)
        encoder_features = features["out"]
        decoder_out_features = self.classifier(encoder_features)
        x = F.interpolate(decoder_out_features, size=input_size, mode="bilinear", align_corners=False)
    
        return encoder_features,decoder_out_features, x


def load_deeplabv3():
    model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False) 
    model.classifier[-1] = nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    ) 
    
    # deeplab_model = DeepLabV3WithFeatures(model)
    
    deeplab_model = model
    
    checkpoint_path= "/scratch/st-puranga-1/users/bassant/code/cv_project/best.pt"
    # "/scratch/st-puranga-1/users/bassant/code/cv_project/mdeeplabv3_resnet50_random.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    deeplab_model.load_state_dict(filtered_state_dict, strict=False)
    
    
    for param in deeplab_model.parameters():
        param.requires_grad = False
    
      
    return deeplab_model
    
    
    
if __name__ =="__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    def overlay(frame, mask, save_path=None, alpha=0.5):
        """
        Overlay a mask on a frame and display or save it.

        Args:
            frame: (3, H, W) tensor or array (assumed to be in range [0, 1] or [0, 255])
            mask: (H, W) binary mask (0 or 255)
            save_path: where to save the image, if any
            alpha: transparency for mask overlay
        """
        # Convert to HWC for matplotlib if needed
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Normalize frame if needed
        if frame.max() > 1:
            frame = frame / 255.0

        frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)

        # Create red mask overlay
        # red_mask = np.zeros_like(frame)
        # red_mask[..., 0] = mask / 255.0  # Red channel only

        # overlay = (1 - alpha) * frame + alpha * red_mask
        overlay = frame.copy()
        red_area = mask > 0
        overlay[red_area] = (
            (1 - alpha) * frame[red_area] + alpha * np.array([1, 0, 0])  # Red
        )


        plt.figure(figsize=(3, 3))
        plt.imshow(overlay)
        plt.axis('off')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def apply_mask_overlay(video, mask, alpha=0.5):
        """
        Apply a red mask overlay to an RGB video using alpha blending.

        Args:
            video: (T, 3, H, W) NumPy array or tensor
            mask:  (T, H, W) binary mask array (0 or 255 or bool)
            alpha: blending factor

        Returns:
            overlay_video: (T, 3, H, W) NumPy array with red overlays where mask > 0
        """
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        if video.max() > 1:
            video = video / 255.0  # Normalize to [0, 1]

        video_overlay = video.copy()  # Avoid modifying original

        for t in range(video.shape[0]):
            frame = video[t]  # (3, H, W)
            red_mask = np.zeros_like(frame)  # (3, H, W)
            red_mask[0] = (mask[t] > 0).astype(np.float32)  # Red channel only

            # Alpha blend only where mask > 0
            blend = (1 - alpha) * frame + alpha * red_mask
            video_overlay[t] = np.where(mask[t][None, :, :] > 0, blend, frame)

        return video_overlay

    def draw_mask(image, binary_mask,save_path, color=(0, 0, 255), alpha=0.5):
        # image = image.squeeze(0)

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, 3)

        if isinstance(binary_mask, torch.Tensor):
            binary_mask = binary_mask.cpu().numpy()

        # if image.shape[0] == 3:  
        #     image = image[0]  # Take the first channel (same across all channels)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        
        # Ensure binary_mask is binary
        # binary_mask = (binary_mask > 0).astype(np.uint8)
        # binary_mask = binary_mask.transpose(1, 2, 0) #(H, W, 1)
        

        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            #   cv2.COLOR_RGB2BGR)
        binary_mask = binary_mask.squeeze(0)
        binary_mask = (binary_mask > 0).astype(np.uint8)

        print(f"original_image_shape {image.shape}, bimary_mask :{binary_mask.shape}")
        print(np.unique(binary_mask))
        # Blend the mask with the image
        # masked_image = cv2.addWeighted(image, 1, mask_colored, alpha, 0)
        
        # Plot original and masked images side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # image = np.clip(image, 0, 255).astype(np.uint8)

        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(binary_mask, cmap="gray") 
        ax[1].set_title("output  mask")
        ax[1].axis("off")

        # Save the plot
        
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Close the figure to free memory

    def savevideo(filename, array, fps):
        """Saves a video to a file.

        Args:
            filename (str): filename of video
            array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
            fps (float or int): frames per second

        Returns:
            None
        """

        c, _, height, width = array.shape

        if c != 3:
            raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in array.transpose((1, 2, 3, 0)):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
    
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #  edit_here
        v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
        # v = np.zeros((frame_height, frame_width, 3, frame_count), dtype=np.uint8)
        # v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            #  edit_here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count, :, :] = frame
        

        v = v.transpose((3, 0, 1, 2))

        return v
    
    # H,W,3,F
    # mea_f, std_f = [0.1289, 0.1289, 0.1289], [0.1911, 0.1911, 0.1911]
    # (channels=3, frames, height, width)
    cine_vid = _loadvideo("/arc/project/st-puranga-1/datasets/echonet/Videos/0X1A3D565B371DC573.avi") # H,W,3,F
    mean=  (32.55479 , 32.68759 , 32.989567)
    std = (9.92539 , 49.93201 , 50.176304)

    trans = Compose([
                        ToTensor(),
                        Normalize(mean, std)
                    ])
    
    
    print(f"video before transformation: {cine_vid.shape}") # C,F,H,W
    

    # Frames = cine_vid.shape[1]
    # print(type(cine_vid)) # numpy 
    # transformed_frames = [trans(cine_vid[:, i, :, :].transpose(1, 2, 0)) for i in range(Frames)]

    # # Stack frames into a 4D tensor (F, 3, H, W)
    # # transformed_frames C,H,W
    # cine_vid = torch.stack(transformed_frames, dim=1)  # Shape: (F, 3, H, W)
    

    # input_tensor = cine_vid # Shape: [3, 174, 112, 112] c,f,h,w
    # print(f"input_tensor: {input_tensor.shape}")
  
    # print(f"video after transformation: {cine_vid.shape}")  # [3, 174, 112, 112]



    deeplab_model = load_deeplabv3()
    deeplab_model.eval()
    print(deeplab_model)
    batch_size = 20
    # x = cine_vid.permute(1,0,2,3)
    x = torch.from_numpy(cine_vid).float()
    x = x.permute(1,0,2,3)  # Change the order to (F,C,H,W)
    # print(f"x shape:{x.shape}")  # F, C, H,W
    # input_tensor shape (T,C,H,W)
  
    # np.concatenate(y_list)
    y = np.concatenate([
    deeplab_model(x[i:(i + batch_size), :, :, :])["out"].detach().cpu().numpy() 
    for i in range(0, x.shape[0], batch_size)
    ])
    

    print(f"y shape {y.shape}")  # F,C,H,W
    # for frame in range(y.shape[0]):
    #     if frame % 10 == 0:
    #         draw_mask(x[frame], y[frame],f"/scratch/st-puranga-1/users/bassant/code/cv_project/debug_images/output_{frame}")
    start = 0
    x = x.numpy()
    # for (i, (filename, offset)) in enumerate(zip(filenames, length)):
    # Extract one video and segmentation predictions
    video = x
    logit = y[:, 0, :, :]
    print(f"out put before plotting ;{logit.shape}, {video.shape}")
    # mask = (logit > 0).astype(np.uint8) * 255  # Binary mask scaled to 0 or 255
    overlay_video = apply_mask_overlay(video, logit)

    for frame_idx in range(0, overlay_video.shape[0], 10):
        frame = overlay_video[frame_idx]
        frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3) for plotting

        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f"/scratch/st-puranga-1/users/bassant/code/cv_project/debug_images/without_transforms_overlay/overlay_frame_{frame_idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    # for frame in range(y.shape[0]):
    #     if frame % 10 == 0:
    #         print(f"before drwaing x : {x.shape} , logit : {y.shape}")
    #         overlay(x[frame], logit[frame],f"/scratch/st-puranga-1/users/bassant/code/cv_project/debug_images/without_transforms/output_{frame}")
    #         # draw_mask(x[frame], mask[frame],f"/scratch/st-puranga-1/users/bassant/code/cv_project/debug_images/without_transforms/output_{frame}")

    # Convert to torch tensors
    std = torch.tensor(std, dtype=torch.float32)
    mean = torch.tensor(mean, dtype=torch.float32)
    video = torch.tensor(video)

    video *= std.reshape(1, 3, 1, 1)
    video += mean.reshape(1, 3, 1, 1)
    # Get frames, channels, height, and width
    f, c, h, w = video.shape  # pylint: disable=W0612
    assert c == 3

    # Put two copies of the video side by side
    video = np.concatenate((video, video), 3)

    # If a pixel is in the segmentation, saturate blue channel
    # Leave alone otherwise
    video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

    # Add blank canvas under pair of videos
    video = np.concatenate((video, np.zeros_like(video)), 2)

    # Compute size of segmentation per frame
    print(f"(logit > 0) : {(logit > 0)}")
    size = (logit > 0).sum((1, 2))

    # # Identify systole frames with peak detection
    # trim_min = sorted(size)[round(len(size) ** 0.05)]
    # trim_max = sorted(size)[round(len(size) ** 0.95)]
    # trim_range = trim_max - trim_min
    # systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

    # Write sizes and frames to file
    # for (frame, s) in enumerate(size):
    #     g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

    # # Plot sizes
    # fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
    # plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
    # ylim = plt.ylim()
    # for s in systole:
    #     plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
    # plt.ylim(ylim)
    # plt.title(os.path.splitext("file.avi")[0])
    # plt.xlabel("Seconds")
    # plt.ylabel("Size (pixels)")
    # plt.tight_layout()
    output ="/scratch/st-puranga-1/users/bassant/code/cv_project/seg_debug/new_trial/"
    # plt.savefig(os.path.join(output, "size"+ ".pdf"))
    # plt.close(fig)

    # # # Normalize size to [0, 1]
    size -= size.min()
    size = size / size.max()
    size = 1 - size

    # Iterate the frames in this video
    print(f"size is : {size}")
    # # for (f, s) in enumerate(size):
    #     print(f"f :{f}, s :{s}")

    #     # On all frames, mark a pixel for the size of the frame
    #     video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

    #     if f in systole:
    #         # If frame is computer-selected systole, mark with a line
    #         video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

    #     def dash(start, stop, on=10, off=10):
    #         buf = []
    #         x = start
    #         while x < stop:
    #             buf.extend(range(x, x + on))
    #             x += on
    #             x += off
    #         buf = np.array(buf)
    #         buf = buf[buf < stop]
    #         return buf
    #     d = dash(115, 224)

    #     large_index , small_index = 176,152
    #     if f == large_index:
    #         # If frame is human-selected diastole, mark with green dashed line on all frames
    #         video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
    #     if f == small_index:
    #         # If frame is human-selected systole, mark with red dashed line on all frames
    #         video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

    #     # Get pixels for a circle centered on the pixel
    #     r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

    #     # On the frame that's being shown, put a circle over the pixel
    #     video[f, :, r, c] = 255.

    # Rearrange dimensions and save
    video = video.transpose(1, 0, 2, 3) # FCHW > CFHW
    video = video.astype(np.uint8)
    savevideo(os.path.join(output, "withou_transformers_new_trial_seg_0X1A3D565B371DC573.avi"), video, 50)
    



    
    # # B =1
    # Frames = input_tensor.shape[1]
    # # for b in range(B):  
    # for f in range(Frames):  
    #     frame = input_tensor[:,f,:,:]  # Shape (C,1, H, W)

    #     frame = frame.squeeze(1) # (C,H,W)
    #     # Add batch dimension (1, C, H, W)
    #     frame = frame.unsqueeze(0) 

    #     encoder_features, decoder_features, final_output = deeplab_model(frame)

    #     print("Encoder features shape:", encoder_features.shape)  # Check encoder features shape ([1, 2048, 14, 14])
    #     print("decoder features shape:", decoder_features[0].shape)  # Check decoder features shape  ([1, 2048, 14, 14])
    #     print("Final output shape:", final_output.shape)  # Check final output shape [1, 1, 112, 112]
        
    #     # sigmoid_output = torch.sigmoid(final_output)
    #     # mask = sigmoid_output.squeeze(1).detach().cpu().numpy()  # Shape: (frames, h, w)
    #     # mask = final_output.cpu().numpy()  # Shape: (112, 112)
    #     # mask = (mask > 0.5).astype(int)  # Convert logits to binary mask (thresholding)
    #     # mask = torch.sigmoid(final_output).squeeze().cpu().numpy()  # Apply sigmoid
    #     mask = final_output.squeeze().cpu().numpy() 
    #     binary_mask = (mask > 0).astype(int)  # Convert probabilities to binary mask
    #     # binary_mask = mask
    #     # binary_mask = (mask > 0.5).astype(np.uint8)
    #     print(np.unique(mask))
    #     print(np.unique(binary_mask))
        
    #     save_path = f"seg_debug/thr_0_original_&_mask_frame{f}.png"
    #     print(f"binary_mask:{binary_mask.shape}")  # (1, 112, 112) >(112,112)
    #     draw_mask(frame, binary_mask, save_path)



    # # for i, ef in enumerate(encoder_features):
    # #     print(f"Encoder feature {i} shape: {ef.shape}")
    
    # # for i, df in enumerate(decoder_features): 
    # #     print(f"decoder feature {i} shape: {df.shape}")

    


