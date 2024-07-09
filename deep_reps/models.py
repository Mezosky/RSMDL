import open_clip
import torch
from PIL import Image


class CLIPAndTokenizerLayers(torch.nn.Module):
    def __init__(
        self, model_name="ViT-B-32", pretrained="laion400m_e32", device="cpu", seed=0
    ):
        super().__init__()
        self.model, self.preprocess, self.tokenizer = (
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self._set_seed(seed)
        self.model.eval()

        self.img_features = {}
        self.text_features = {}

        def get_activations(name, feature_dict):
            def hook(model, input, output):
                feature_dict[name] = output

            return hook

        for name, module in self.model.visual.named_modules():
            module.register_forward_hook(get_activations(name, self.img_features))

        for name, module in self.model.transformer.named_modules():
            module.register_forward_hook(get_activations(name, self.text_features))

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _load_image_tensor(self, image_path):
        img_loaded = self.preprocess(Image.open(image_path)).unsqueeze(0)
        return img_loaded.to(self.device)

    def forward(self, img_path, text_tokens):
        img = self._load_image_tensor(img_path)
        with torch.no_grad():
            _ = self.model.encode_image(img)

        if text_tokens:
            text = self.tokenizer(text_tokens).to(self.device)
            with torch.no_grad():
                _ = self.model.encode_text(text)
            return self.img_features, self.text_features

        return self.img_features
