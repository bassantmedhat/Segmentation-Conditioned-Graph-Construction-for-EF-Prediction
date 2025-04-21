import sys
sys.path.append('sam2')

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import torch


def concat_all_outputs(inference_state):
    pred_masks_list = []
    maskmem_features_list = []

    for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
        frame_outputs = inference_state["output_dict"][storage_key]
        for frame_idx in sorted(frame_outputs.keys()):
            out = frame_outputs[frame_idx]

            if out.get("pred_masks") is not None:
                pred_masks_list.append(out["pred_masks"])  # (B, 1, H, W)
            
            if out.get("maskmem_features") is not None:
                maskmem_features_list.append(out["maskmem_features"])  # (B, C, H, W)

    if pred_masks_list:
        all_pred_masks = torch.cat(pred_masks_list, dim=0)  # (T*B, 1, H, W)
    else:
        all_pred_masks = None

    if maskmem_features_list:
        all_maskmem_features = torch.cat(maskmem_features_list, dim=0)  # (T*B, C, H, W)
    else:
        all_maskmem_features = None

    return all_pred_masks, all_maskmem_features

def run_sam(x, base_video_dir,input_mask_dir,output_mask_dir,video_name, count_save):
    # helper functions
    DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

    def load_ann_png(path):
        """Load a PNG file as a mask and its palette."""
        mask = Image.open(path)
        palette = mask.getpalette()
        mask = np.array(mask).astype(np.uint8)
        return mask, palette

    def get_per_obj_mask(mask):
        """Split a mask into per-object masks."""
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids > 0].tolist()
        per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
        return per_obj_mask

    def put_per_obj_mask(per_obj_mask, height, width):
        """Combine per-object masks into a single mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        object_ids = sorted(per_obj_mask)[::-1]
        for object_id in object_ids:
            object_mask = per_obj_mask[object_id]
            object_mask = object_mask.reshape(height, width)
            mask[object_mask] = object_id
        return mask

    def load_masks_from_dir(input_mask_path):
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)

        return per_obj_input_mask, input_palette

    def save_predictions_to_dir(
        output_mask_dir,
        video_name,
        frame_name,
        per_obj_output_mask,
        height,
        width,
    ):
        """Save masks to a directory as PNG files."""
        os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        assert output_mask.dtype == np.uint8
        assert output_mask.ndim == 2
        output_mask = Image.fromarray(output_mask)
        output_mask.save(output_mask_path)

    def create_overlay(img_path, mask_path, palette):
        """Create an overlay of an image and a mask."""
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask.putpalette(palette)
        mask_rgb = mask.convert("RGB")
        mask_rgb = mask_rgb.resize(img.size, Image.NEAREST)
        overlay = Image.blend(img, mask_rgb, alpha=0.5)
        return overlay
        


    # change to customized path
    VIDEO_DIR = base_video_dir
    VIDEO_NAME = video_name
    INITIAL_MASK_PROMPT = input_mask_dir
    OUTPUT_DIR = output_mask_dir

    MODEL_CONFIG = "configs/sam2.1_hiera_t512.yaml"
    MODEL_CHECKPOINT = "MedSAM2_latest.pt"

    predictor = build_sam2_video_predictor(
        config_file=MODEL_CONFIG,
        ckpt_path=MODEL_CHECKPOINT,
        apply_postprocessing=True,
        # hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=  True,
    )

    frame_names = list(VIDEO_NAME)
    frame_names = list(sorted(frame_names))
    inference_state = predictor.init_state(
        x, video_path=os.path.join(VIDEO_DIR, VIDEO_NAME), async_loading_frames=False
    )
    # print(type(inference_state["images"]),inference_state["images"].shape )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # Add input masks to MedSAM2 inference state before propagation
    object_ids_set = None
    input_frame_idx = 0  # use first frame as mask input
    try:
        per_obj_input_mask, input_palette = load_masks_from_dir(input_mask_path=INITIAL_MASK_PROMPT)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"In {VIDEO_NAME=}, failed to load input mask for frame {input_frame_idx=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        ) from e

    # get the list of object ids to track from the first input frame
    if object_ids_set is None:
        object_ids_set = set(per_obj_input_mask)
    for object_id, object_mask in per_obj_input_mask.items():
        # check and make sure no new object ids appear only in later frames
        if object_id not in object_ids_set:
            raise RuntimeError(
                f"In {VIDEO_NAME=}, got a new {object_id=} appearing only in a "
                f"later {input_frame_idx=} (but not appearing in the first frame). "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            )
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_id=object_id,
            mask=object_mask,
                )
        
    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {VIDEO_NAME=}, got no object ids on {input_frame_idx=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )



    frame_visual_features = []
    frame_mask_features = []
    video_segments = {}
    itr = 0 

    # loop over each frame 
    for idx, (out_frame_idx, out_obj_ids, out_mask_logits, visual_features, segmentation_features) in enumerate(
        predictor.propagate_in_video(inference_state)
    ):


        if len(visual_features) == 0:
            continue

        # print(f" idx and len of visual_features : {idx - itr} , {len(visual_features)}")


        if visual_features[0] !=  None:
            # remove.cpu() here 
            # remove idx to be 0 
            frame_visual_features.append(visual_features[0])
            frame_mask_features.append(visual_features[0])

            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output_mask

    # stack safely on CPU
    finalframe_visual_features = torch.stack(frame_visual_features)
    finalframe_mask_features = torch.stack(frame_mask_features)

    # cleanup
    del frame_visual_features, frame_mask_features
    torch.cuda.empty_cache()


    if finalframe_visual_features.shape[0] == 31 or finalframe_visual_features.shape[0] == 63:
        # Create an empty (zero) tensor of the same shape as one frame
        padding = torch.zeros((1, 1, 256, 16, 16), dtype=finalframe_visual_features.dtype, device=finalframe_visual_features.device)

        # Concatenate along the 0th dimension (channels/frames)
        finalframe_visual_features = torch.cat([finalframe_visual_features, padding], dim=0)
        finalframe_mask_features = torch.cat([finalframe_mask_features, padding], dim=0)

    # write the output masks
    if count_save:
        os.makedirs(os.path.join(OUTPUT_DIR, VIDEO_NAME), exist_ok=True)
        for out_frame_idx, per_obj_output_mask in video_segments.items():

            save_predictions_to_dir(
                output_mask_dir=OUTPUT_DIR,
                video_name=VIDEO_NAME,
                frame_name=frame_names[0],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
            )




    return finalframe_visual_features, finalframe_mask_features



# if __name__== '__main__':
#     video_segments, inference_state = run_sam()
#     print(f"video_segments:{video_segments.items()}")
#     print(f"inference_state:{inference_state}")

