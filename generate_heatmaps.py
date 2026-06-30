import argparse
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from visualization import visualizer

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)

    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    model.to(device)

    target_classes = set(args.classes) if args.classes else None
    saved_count = 0

    for items in test_dataloader:
        cls_name = items['cls_name']
        if target_classes and cls_name[0] not in target_classes:
            continue
        if args.max_per_class is not None and saved_count >= args.max_per_class * (len(target_classes) if target_classes else 1):
            break

        image = items['img'].to(device)
        img_path = items['img_path']

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()],
                dim=0
            )

        visualizer(img_path, anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
        saved_count += 1
        print(f"Saved heatmap {saved_count}: {cls_name[0]} - {img_path[0].split('/')[-1]}")

    print(f"\nDone. Saved {saved_count} heatmaps to {args.save_path}/imgs/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP heatmap generator")
    parser.add_argument("--data_path", type=str, default="./data/mvtec_ad")
    parser.add_argument("--save_path", type=str, default="./results/heatmaps")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/9_12_4_multiscale_visa/epoch_15.pth")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="e.g. --classes leather")
    parser.add_argument("--max_per_class", type=int, default=None, help="limit images per class")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sigma", type=int, default=4)
    args = parser.parse_args()
    main(args)
