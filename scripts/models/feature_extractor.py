import torch

DEFAULT_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_cow_features(img, bbox, feature_extractor, device="cuda", normalize=True, input_size=DEFAULT_INPUT_SIZE):
    """Crop `bbox` out of `img` and return the hooked feature vector.

    cv2 / torchvision are imported here rather than at module scope so that the
    caching and pipeline-orchestration logic stays importable on a box without
    the vision stack; the import cost is a sys.modules hit next to a forward pass.
    `input_size` is threaded through instead of hardcoded because the feature
    cache is keyed on it — a hardcoded resize would make the cache key lie.
    """
    import cv2
    import torchvision.transforms as transforms

    x1_orig, y1_orig, x2_orig, y2_orig = bbox

    h, w = img.shape[:2]
    x1 = max(0, round(x1_orig))
    y1 = max(0, round(y1_orig))
    x2 = min(w, round(x2_orig))
    y2 = min(h, round(y2_orig))

    if x2 <= x1:
        x2 = x1 + 1
        if x2 > w:
            x2 = w
            x1 = w - 1
            if x1 < 0:
                return None
    if y2 <= y1:
        y2 = y1 + 1
        if y2 > h:
            y2 = h
            y1 = h - 1
            if y1 < 0:
                return None

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x1 >= x2 or y1 >= y2:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (input_size, input_size))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    transform = transforms.Compose(transform_list)
    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = feature_extractor(input_tensor)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    if feat.dim() > 2:
        feat = feat.mean(dim=[2, 3])
    feature_vector = feat.cpu().numpy().flatten()

    return feature_vector


def create_feature_extractor(yolo_model_path, layer_index=9, device="cuda"):
    from ultralytics import YOLO

    model = YOLO(yolo_model_path)
    pt_model = model.model
    pt_model.to(device)
    pt_model.eval()
    features = []

    def hook_fn(module, inp, out):
        features.append(out.detach())

    hook = pt_model.model[layer_index].register_forward_hook(hook_fn)

    def extractor(x):
        features.clear()
        _ = pt_model(x)
        feat_map = features[0]
        feat_vec = feat_map.mean(dim=[2, 3]).squeeze()
        return feat_vec

    return extractor, hook
